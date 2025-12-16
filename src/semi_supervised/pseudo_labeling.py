import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import gc


class PseudoLabeler:
    def __init__(self, config, confidence_thresholds=None):
        self.config = config
        self.confidence_thresholds = confidence_thresholds or {}
        self.default_threshold = config.PSEUDO_LABEL_CONFIDENCE

    def generate_pseudo_labels(
        self,
        classifier,
        unlabeled_features: pd.DataFrame,
        unlabeled_video_ids: pd.Series,
        actions: List[str],
        per_action_threshold: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        print(f"\nGenerating pseudo-labels for {len(unlabeled_features)} samples")
        print(f"Actions: {len(actions)}")

        predictions = classifier.predict_multiple_actions(unlabeled_features, actions)

        pseudo_labels = pd.DataFrame(0, index=unlabeled_features.index, columns=actions)
        confidence_scores = pd.DataFrame(0.0, index=unlabeled_features.index, columns=actions)

        selected_indices = []

        for action in actions:
            if action not in predictions.columns:
                continue

            probs = predictions[action].values

            threshold = self.confidence_thresholds.get(action, self.default_threshold)

            high_conf_mask = probs >= threshold
            low_conf_mask = probs <= (1 - threshold)

            pseudo_labels.loc[high_conf_mask, action] = 1
            pseudo_labels.loc[low_conf_mask, action] = 0
            confidence_scores[action] = probs

            selected = high_conf_mask | low_conf_mask
            selected_indices.append(selected)

        if per_action_threshold:
            combined_mask = np.any(selected_indices, axis=0)
        else:
            combined_mask = np.all(selected_indices, axis=0)

        n_selected = combined_mask.sum()
        selection_rate = n_selected / len(unlabeled_features) * 100

        print(f"Selected {n_selected}/{len(unlabeled_features)} samples ({selection_rate:.1f}%)")

        pseudo_features = unlabeled_features[combined_mask]
        pseudo_labels_filtered = pseudo_labels[combined_mask]
        pseudo_video_ids = unlabeled_video_ids[combined_mask]

        return pseudo_features, pseudo_labels_filtered, pseudo_video_ids

    def adaptive_threshold_update(
        self,
        action_f1_scores: Dict[str, float],
        min_threshold: float = 0.7,
        max_threshold: float = 0.95
    ):

        print("\nUpdating confidence thresholds based on performance...")

        for action, f1 in action_f1_scores.items():
            if f1 > 0.7:
                new_threshold = max(min_threshold, self.default_threshold - 0.05)
            elif f1 < 0.4:
                new_threshold = min(max_threshold, self.default_threshold + 0.05)
            else:
                new_threshold = self.default_threshold

            old_threshold = self.confidence_thresholds.get(action, self.default_threshold)
            self.confidence_thresholds[action] = new_threshold

            if abs(new_threshold - old_threshold) > 0.01:
                print(f"  {action}: {old_threshold:.2f} -> {new_threshold:.2f} (F1={f1:.3f})")

    def filter_noisy_labels(
        self,
        pseudo_labels: pd.DataFrame,
        predictions: pd.DataFrame,
        consistency_threshold: float = 0.9
    ) -> pd.Series:

        consistency_scores = []

        for action in pseudo_labels.columns:
            if action not in predictions.columns:
                continue

            label = pseudo_labels[action]
            pred_prob = predictions[action]

            consistent = ((label == 1) & (pred_prob > consistency_threshold)) | \
                        ((label == 0) & (pred_prob < (1 - consistency_threshold)))

            consistency_scores.append(consistent)

        if len(consistency_scores) == 0:
            return pd.Series(True, index=pseudo_labels.index)

        consistency_mask = pd.concat(consistency_scores, axis=1).all(axis=1)

        n_filtered = (~consistency_mask).sum()
        print(f"Filtered {n_filtered} inconsistent samples")

        return consistency_mask

    def balance_pseudo_labels(
        self,
        pseudo_features: pd.DataFrame,
        pseudo_labels: pd.DataFrame,
        pseudo_video_ids: pd.Series,
        max_ratio: float = 5.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        print("\nBalancing pseudo-labeled data...")

        balanced_features = []
        balanced_labels = []
        balanced_video_ids = []

        for action in pseudo_labels.columns:
            pos_mask = pseudo_labels[action] == 1
            neg_mask = pseudo_labels[action] == 0

            n_pos = pos_mask.sum()
            n_neg = neg_mask.sum()

            if n_pos == 0 or n_neg == 0:
                continue

            ratio = n_neg / n_pos

            if ratio > max_ratio:
                n_neg_sample = int(n_pos * max_ratio)
                neg_indices = np.random.choice(
                    np.where(neg_mask)[0],
                    size=n_neg_sample,
                    replace=False
                )
                keep_mask = pos_mask.copy()
                keep_mask.iloc[neg_indices] = True
            else:
                keep_mask = pos_mask | neg_mask

            print(f"  {action}: {n_pos} pos, {n_neg} neg (ratio: {ratio:.1f})")

        return pseudo_features, pseudo_labels, pseudo_video_ids

    def iterative_pseudo_labeling(
        self,
        classifier,
        labeled_data: Dict,
        unlabeled_data: Dict,
        actions: List[str],
        n_iterations: int = 3
    ) -> Dict:

        print(f"\n{'='*60}")
        print(f"ITERATIVE PSEUDO-LABELING: {n_iterations} iterations")
        print(f"{'='*60}")

        current_labeled = labeled_data.copy()

        for iteration in range(1, n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{n_iterations}")
            print(f"{'='*60}")

            print(f"\nCurrent labeled size: {len(current_labeled['features'])}")

            pseudo_features, pseudo_labels, pseudo_video_ids = self.generate_pseudo_labels(
                classifier,
                unlabeled_data['features'],
                unlabeled_data['video_ids'],
                actions
            )

            if len(pseudo_features) == 0:
                print("No pseudo-labels generated, stopping")
                break

            combined_features = pd.concat([
                current_labeled['features'],
                pseudo_features
            ], axis=0, ignore_index=True)

            combined_labels = pd.concat([
                current_labeled['labels'],
                pseudo_labels
            ], axis=0, ignore_index=True)

            combined_video_ids = pd.concat([
                current_labeled['video_ids'],
                pseudo_video_ids
            ], axis=0, ignore_index=True)

            print(f"\nRetraining with {len(combined_features)} samples...")
            print(f"  Original: {len(current_labeled['features'])}")
            print(f"  Pseudo: {len(pseudo_features)}")

            f1_scores = classifier.train_multiple_actions(
                combined_features,
                combined_labels,
                combined_video_ids,
                actions,
                f"pseudo_iter_{iteration}"
            )

            if iteration < n_iterations:
                self.adaptive_threshold_update(f1_scores)

            current_labeled = {
                'features': combined_features,
                'labels': combined_labels,
                'video_ids': combined_video_ids
            }

            gc.collect()

        print(f"\n{'='*60}")
        print("Pseudo-labeling complete!")
        print(f"Final dataset size: {len(current_labeled['features'])}")
        print(f"{'='*60}")

        return current_labeled
