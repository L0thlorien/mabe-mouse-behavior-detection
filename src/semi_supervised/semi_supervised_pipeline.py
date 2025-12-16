import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import gc
import torch

from ..data_loader import DataLoader, DataProcessor
from ..features import FeatureEngineer
from ..models import BehaviorClassifier
from .autoencoder import AutoencoderTrainer
from .pseudo_labeling import PseudoLabeler


class SemiSupervisedPipeline:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self) -> Dict[str, float]:
        print("="*60)
        print("SEMI-SUPERVISED TRAINING PIPELINE")
        print("="*60)
        print(f"Device: {self.device}")

        print("\n1. Loading metadata...")
        all_train_df = self.loader.load_metadata(mode="train")

        labeled_df = all_train_df[~all_train_df['lab_id'].str.startswith('MABe22')].reset_index(drop=True)
        unlabeled_df = all_train_df[all_train_df['lab_id'].str.startswith('MABe22')].reset_index(drop=True)

        print(f"   Total videos: {len(all_train_df)}")
        print(f"   Labeled: {len(labeled_df)}")
        print(f"   Unlabeled: {len(unlabeled_df)}")

        body_parts_groups = labeled_df.groupby('body_parts_tracked')
        print(f"   Body part configurations: {len(body_parts_groups)}")

        all_f1_scores = {}

        for section_id, (body_parts_str, group_df) in enumerate(body_parts_groups):
            print(f"\n{'='*60}")
            print(f"Section {section_id + 1}/{len(body_parts_groups)}")
            print(f"Body parts: {body_parts_str[:100]}...")
            print(f"Videos: {len(group_df)}")
            print(f"{'='*60}")

            unlabeled_section = unlabeled_df[
                unlabeled_df['body_parts_tracked'] == body_parts_str
            ].reset_index(drop=True)

            section_scores = self._train_section_semi_supervised(
                group_df,
                unlabeled_section,
                section_id
            )
            all_f1_scores.update(section_scores)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        if len(all_f1_scores) > 0:
            mean_f1 = np.mean(list(all_f1_scores.values()))
            print(f"Mean F1 Score: {mean_f1:.4f}")
            print(f"Total actions trained: {len(all_f1_scores)}")
        else:
            print("No models trained")

        return all_f1_scores

    def _train_section_semi_supervised(
        self,
        labeled_section_df: pd.DataFrame,
        unlabeled_section_df: pd.DataFrame,
        section_id: int
    ) -> Dict[str, float]:

        print("\n" + "="*60)
        print("STAGE 1: AUTOENCODER PRETRAINING")
        print("="*60)

        all_features = self._extract_features_for_videos(
            pd.concat([labeled_section_df, unlabeled_section_df]),
            "all videos"
        )

        if len(all_features) == 0:
            print("No features extracted, skipping section")
            return {}

        labeled_features = self._extract_features_for_videos(
            labeled_section_df,
            "labeled videos"
        )

        n_features = all_features[0].shape[1] if len(all_features) > 0 else 0

        if self.config.USE_AUTOENCODER and n_features > 0:
            autoencoder_trainer = AutoencoderTrainer(
                self.config,
                n_features=n_features,
                device=self.device
            )

            train_size = int(len(all_features) * 0.9)
            train_features = all_features[:train_size]
            val_features = all_features[train_size:]

            autoencoder_trainer.train(train_features, val_features)

            print("\nExtracting embeddings for labeled data...")
            labeled_embeddings = autoencoder_trainer.extract_embeddings(labeled_features)
        else:
            labeled_embeddings = None

        print("\n" + "="*60)
        print("STAGE 2: PROCESSING LABELED DATA")
        print("="*60)

        single_data = {'features': [], 'labels': [], 'video_ids': []}
        pair_data = {'features': [], 'labels': [], 'video_ids': []}

        embed_idx = 0
        for idx, row in tqdm(labeled_section_df.iterrows(), total=len(labeled_section_df), desc="Labeled"):
            video_info = self.loader.get_video_info(row)

            if len(video_info['behaviors_labeled']) == 0:
                continue

            try:
                result = self._process_video_with_labels(video_info, labeled_embeddings, embed_idx)
                if result:
                    single_data, pair_data = self._merge_video_data(result, single_data, pair_data)
                    embed_idx += 1
            except Exception as e:
                print(f"\nError processing {video_info['video_id']}: {e}")
                continue

        print("\n" + "="*60)
        print("STAGE 3: PROCESSING UNLABELED DATA")
        print("="*60)

        unlabeled_features = self._extract_features_for_videos(
            unlabeled_section_df,
            "unlabeled videos"
        )

        if self.config.USE_AUTOENCODER and len(unlabeled_features) > 0:
            print("\nExtracting embeddings for unlabeled data...")
            unlabeled_embeddings = autoencoder_trainer.extract_embeddings(unlabeled_features)
        else:
            unlabeled_embeddings = None

        unlabeled_single = {'features': [], 'video_ids': []}
        unlabeled_pair = {'features': [], 'video_ids': []}

        embed_idx = 0
        for idx, row in tqdm(unlabeled_section_df.iterrows(), total=len(unlabeled_section_df), desc="Unlabeled"):
            video_info = self.loader.get_video_info(row)

            try:
                result = self._process_video_without_labels(video_info, unlabeled_embeddings, embed_idx)
                if result:
                    unlabeled_single, unlabeled_pair = self._merge_unlabeled_data(
                        result, unlabeled_single, unlabeled_pair
                    )
                    embed_idx += 1
            except Exception as e:
                continue

        print("\n" + "="*60)
        print("STAGE 4: TRAINING WITH SEMI-SUPERVISED LEARNING")
        print("="*60)

        f1_scores = {}

        if len(single_data['features']) > 0:
            print("\nTraining SINGLE behavior classifiers...")
            single_scores = self._train_with_pseudo_labeling(
                single_data,
                unlabeled_single,
                section_id,
                'single'
            )
            f1_scores.update(single_scores)

        if len(pair_data['features']) > 0:
            print("\nTraining PAIR behavior classifiers...")
            pair_scores = self._train_with_pseudo_labeling(
                pair_data,
                unlabeled_pair,
                section_id,
                'pair'
            )
            f1_scores.update(pair_scores)

        return f1_scores

    def _extract_features_for_videos(self, df: pd.DataFrame, desc: str) -> List[np.ndarray]:
        features_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {desc}"):
            video_info = self.loader.get_video_info(row)

            try:
                tracking_df = self.loader.load_tracking_data(
                    video_info['lab_id'],
                    video_info['video_id'],
                    mode="train"
                )

                pose_data = self.processor.pivot_tracking_data(
                    tracking_df,
                    video_info['pix_per_cm']
                )

                mouse_ids = (1,)
                features = self.feature_engineer.engineer_features(
                    pose_data, video_info, 'single', mouse_ids
                )

                if len(features) > 0:
                    features_list.append(features.values)

            except Exception:
                continue

        return features_list

    def _process_video_with_labels(self, video_info: Dict, embeddings, embed_idx):
        tracking_df = self.loader.load_tracking_data(
            video_info['lab_id'],
            video_info['video_id'],
            mode="train"
        )

        annotation_df = self.loader.load_annotation_data(
            video_info['lab_id'],
            video_info['video_id']
        )

        if annotation_df is None:
            return None

        pose_data = self.processor.pivot_tracking_data(
            tracking_df,
            video_info['pix_per_cm']
        )

        parsed_behaviors = self.processor.parse_behaviors_labeled(
            video_info['behaviors_labeled']
        )

        labels_dict = self.processor.create_labels_for_video(
            annotation_df,
            video_info,
            len(pose_data)
        )

        result = {'single': [], 'pair': []}

        for (agent, target), actions in parsed_behaviors['single']:
            mouse_id = int(agent.replace('mouse', ''))
            features = self.feature_engineer.engineer_features(
                pose_data, video_info, 'single', (mouse_id,)
            )

            if embeddings is not None and len(features) > 0:
                embed_repeat = np.tile(embeddings[embed_idx], (len(features), 1))
                features_df = pd.DataFrame(embed_repeat, index=features.index,
                                          columns=[f'embed_{i}' for i in range(embed_repeat.shape[1])])
                features = pd.concat([features, features_df], axis=1)

            if len(features) > 0:
                key = (agent, target)
                if key in labels_dict['single']:
                    result['single'].append((features, labels_dict['single'][key], video_info['video_id']))

        for (agent, target), actions in parsed_behaviors['pair']:
            agent_id = int(agent.replace('mouse', ''))
            target_id = int(target.replace('mouse', ''))
            features = self.feature_engineer.engineer_features(
                pose_data, video_info, 'pair', (agent_id, target_id)
            )

            if embeddings is not None and len(features) > 0:
                embed_repeat = np.tile(embeddings[embed_idx], (len(features), 1))
                features_df = pd.DataFrame(embed_repeat, index=features.index,
                                          columns=[f'embed_{i}' for i in range(embed_repeat.shape[1])])
                features = pd.concat([features, features_df], axis=1)

            if len(features) > 0:
                key = (agent, target)
                if key in labels_dict['pair']:
                    result['pair'].append((features, labels_dict['pair'][key], video_info['video_id']))

        return result

    def _process_video_without_labels(self, video_info: Dict, embeddings, embed_idx):
        tracking_df = self.loader.load_tracking_data(
            video_info['lab_id'],
            video_info['video_id'],
            mode="train"
        )

        pose_data = self.processor.pivot_tracking_data(
            tracking_df,
            video_info['pix_per_cm']
        )

        result = {'single': [], 'pair': []}

        mouse_id = 1
        features = self.feature_engineer.engineer_features(
            pose_data, video_info, 'single', (mouse_id,)
        )

        if embeddings is not None and len(features) > 0:
            embed_repeat = np.tile(embeddings[embed_idx], (len(features), 1))
            features_df = pd.DataFrame(embed_repeat, index=features.index,
                                      columns=[f'embed_{i}' for i in range(embed_repeat.shape[1])])
            features = pd.concat([features, features_df], axis=1)

        if len(features) > 0:
            result['single'].append((features, video_info['video_id']))

        if video_info['n_mice'] >= 2:
            features = self.feature_engineer.engineer_features(
                pose_data, video_info, 'pair', (1, 2)
            )

            if embeddings is not None and len(features) > 0:
                embed_repeat = np.tile(embeddings[embed_idx], (len(features), 1))
                features_df = pd.DataFrame(embed_repeat, index=features.index,
                                          columns=[f'embed_{i}' for i in range(embed_repeat.shape[1])])
                features = pd.concat([features, features_df], axis=1)

            if len(features) > 0:
                result['pair'].append((features, video_info['video_id']))

        return result

    def _merge_video_data(self, result, single_data, pair_data):
        for features, labels, video_id in result['single']:
            single_data['features'].append(features)
            single_data['labels'].append(labels)
            single_data['video_ids'].append([video_id] * len(features))

        for features, labels, video_id in result['pair']:
            pair_data['features'].append(features)
            pair_data['labels'].append(labels)
            pair_data['video_ids'].append([video_id] * len(features))

        return single_data, pair_data

    def _merge_unlabeled_data(self, result, unlabeled_single, unlabeled_pair):
        for features, video_id in result['single']:
            unlabeled_single['features'].append(features)
            unlabeled_single['video_ids'].append([video_id] * len(features))

        for features, video_id in result['pair']:
            unlabeled_pair['features'].append(features)
            unlabeled_pair['video_ids'].append([video_id] * len(features))

        return unlabeled_single, unlabeled_pair

    def _train_with_pseudo_labeling(
        self,
        labeled_data: Dict,
        unlabeled_data: Dict,
        section_id: int,
        behavior_type: str
    ) -> Dict[str, float]:

        X_labeled = pd.concat(labeled_data['features'], axis=0, ignore_index=True)
        y_labeled = pd.concat(labeled_data['labels'], axis=0, ignore_index=True)
        video_ids_labeled = pd.Series(np.concatenate(labeled_data['video_ids']))

        print(f"   Labeled data shape: {X_labeled.shape}")
        print(f"   Actions: {list(y_labeled.columns)}")

        classifier = BehaviorClassifier(self.config)
        print("\nInitial training on labeled data...")
        f1_scores = classifier.train_multiple_actions(
            X_labeled, y_labeled, video_ids_labeled,
            list(y_labeled.columns),
            f"{section_id}_{behavior_type}_initial"
        )

        if not self.config.USE_PSEUDO_LABELING or len(unlabeled_data['features']) == 0:
            print("\nPseudo-labeling disabled or no unlabeled data")
            return f1_scores

        X_unlabeled = pd.concat(unlabeled_data['features'], axis=0, ignore_index=True)
        video_ids_unlabeled = pd.Series(np.concatenate(unlabeled_data['video_ids']))

        print(f"\n   Unlabeled data shape: {X_unlabeled.shape}")

        pseudo_labeler = PseudoLabeler(self.config)

        current_labeled = {
            'features': X_labeled,
            'labels': y_labeled,
            'video_ids': video_ids_labeled
        }

        current_unlabeled = {
            'features': X_unlabeled,
            'video_ids': video_ids_unlabeled
        }

        final_data = pseudo_labeler.iterative_pseudo_labeling(
            classifier,
            current_labeled,
            current_unlabeled,
            list(y_labeled.columns),
            n_iterations=self.config.PSEUDO_LABEL_ITERATIONS
        )

        print("\nFinal training on combined data...")
        final_f1_scores = classifier.train_multiple_actions(
            final_data['features'],
            final_data['labels'],
            final_data['video_ids'],
            list(y_labeled.columns),
            f"{section_id}_{behavior_type}"
        )

        return final_f1_scores
