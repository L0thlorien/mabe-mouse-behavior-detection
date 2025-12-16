import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import kneighbors_graph
from typing import Dict, Tuple
from tqdm import tqdm


class SemiSupervisedLabelPropagation:
    def __init__(self, config, method='spreading'):
        self.config = config
        self.method = method
        self.n_neighbors = 7
        self.alpha = 0.2
        self.max_iter = 30

    def propagate_labels(
        self,
        labeled_features: pd.DataFrame,
        labeled_labels: pd.DataFrame,
        unlabeled_features: pd.DataFrame,
        action: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        if action not in labeled_labels.columns:
            return np.array([]), np.array([])

        y_labeled = labeled_labels[action].values
        n_positive = y_labeled.sum()
        n_negative = len(y_labeled) - n_positive

        if n_positive < 5:
            print(f"  {action}: Too few positive examples ({n_positive}), skipping")
            return np.array([]), np.array([])

        X_combined = np.vstack([
            labeled_features.values,
            unlabeled_features.values
        ])

        y_combined = np.concatenate([
            y_labeled,
            np.full(len(unlabeled_features), -1)
        ])

        n_labeled = len(labeled_features)
        n_unlabeled = len(unlabeled_features)
        n_total = len(X_combined)

        print(f"  {action}: Propagating {n_positive} positives, {n_negative} negatives to {n_unlabeled} unlabeled samples")

        if self.method == 'spreading':
            model = LabelSpreading(
                kernel='knn',
                n_neighbors=self.n_neighbors,
                alpha=self.alpha,
                max_iter=self.max_iter
            )
        else:
            model = LabelPropagation(
                kernel='knn',
                n_neighbors=self.n_neighbors,
                max_iter=self.max_iter
            )

        model.fit(X_combined, y_combined)

        propagated_labels = model.transduction_
        label_distributions = model.label_distributions_

        unlabeled_predictions = propagated_labels[n_labeled:]
        unlabeled_probabilities = label_distributions[n_labeled:, 1]

        return unlabeled_predictions, unlabeled_probabilities

    def propagate_all_actions(
        self,
        labeled_features: pd.DataFrame,
        labeled_labels: pd.DataFrame,
        unlabeled_features: pd.DataFrame,
        actions: list,
        confidence_threshold: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        print(f"\nPropagating labels for {len(actions)} actions...")
        print(f"Labeled samples: {len(labeled_features)}")
        print(f"Unlabeled samples: {len(unlabeled_features)}")

        pseudo_labels = pd.DataFrame(
            0,
            index=unlabeled_features.index,
            columns=actions
        )

        pseudo_probs = pd.DataFrame(
            0.0,
            index=unlabeled_features.index,
            columns=actions
        )

        for action in tqdm(actions, desc="Actions"):
            predictions, probabilities = self.propagate_labels(
                labeled_features,
                labeled_labels,
                unlabeled_features,
                action
            )

            if len(predictions) > 0:
                confident_mask = (probabilities > confidence_threshold) | (probabilities < (1 - confidence_threshold))
                pseudo_labels.loc[confident_mask, action] = predictions[confident_mask]
                pseudo_probs.loc[:, action] = probabilities

        n_samples_with_labels = (pseudo_labels.sum(axis=1) > 0).sum()
        print(f"\nPropagation complete:")
        print(f"  Samples with at least one label: {n_samples_with_labels}/{len(unlabeled_features)}")
        print(f"  Coverage: {n_samples_with_labels/len(unlabeled_features)*100:.1f}%")

        return pseudo_labels, pseudo_probs


def apply_label_propagation_to_section(
    labeled_data: Dict,
    unlabeled_data: Dict,
    config,
    confidence_threshold: float = 0.7
) -> Dict:

    X_labeled = labeled_data['features']
    y_labeled = labeled_data['labels']
    video_ids_labeled = labeled_data['video_ids']

    X_unlabeled = unlabeled_data['features']
    video_ids_unlabeled = unlabeled_data['video_ids']

    actions = list(y_labeled.columns)

    propagator = SemiSupervisedLabelPropagation(config, method='spreading')

    pseudo_labels, pseudo_probs = propagator.propagate_all_actions(
        X_labeled,
        y_labeled,
        X_unlabeled,
        actions,
        confidence_threshold=confidence_threshold
    )

    has_labels = (pseudo_labels.sum(axis=1) > 0)

    if has_labels.sum() == 0:
        print("\nWarning: No pseudo-labels generated")
        return labeled_data

    X_pseudo = X_unlabeled[has_labels]
    y_pseudo = pseudo_labels[has_labels]
    video_ids_pseudo = video_ids_unlabeled[has_labels]

    X_combined = pd.concat([X_labeled, X_pseudo], axis=0, ignore_index=True)
    y_combined = pd.concat([y_labeled, y_pseudo], axis=0, ignore_index=True)
    video_ids_combined = pd.concat([video_ids_labeled, video_ids_pseudo], axis=0, ignore_index=True)

    print(f"\nFinal dataset:")
    print(f"  Original labeled: {len(X_labeled)}")
    print(f"  Pseudo-labeled: {len(X_pseudo)}")
    print(f"  Total: {len(X_combined)}")

    return {
        'features': X_combined,
        'labels': y_combined,
        'video_ids': video_ids_combined
    }
