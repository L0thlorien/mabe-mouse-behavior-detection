import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import gc

from .data_loader import DataLoader, DataProcessor
from .features import FeatureEngineer
from .models import BehaviorClassifier
from .postprocessing import PredictionPostProcessor


class TrainingPipeline:

    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.postprocessor = PredictionPostProcessor(config)

    def run(self) -> Dict[str, float]:
        print("=" * 60)
        print("TRAINING PIPELINE")
        print("=" * 60)

        print("\n1. Loading metadata...")
        train_df = self.loader.load_metadata(mode="train")
        print(f"   Loaded {len(train_df)} training videos")

        body_parts_groups = train_df.groupby('body_parts_tracked')
        print(f"   Found {len(body_parts_groups)} different body part configurations")

        all_f1_scores = {}

        for section_id, (body_parts_str, group_df) in enumerate(body_parts_groups):
            print(f"\n{'=' * 60}")
            print(f"Section {section_id + 1}/{len(body_parts_groups)}")
            print(f"Body parts: {body_parts_str[:100]}...")
            print(f"Videos: {len(group_df)}")
            print(f"{'=' * 60}")

            section_scores = self._train_section(group_df, section_id)
            all_f1_scores.update(section_scores)

            gc.collect()

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        if len(all_f1_scores) > 0:
            mean_f1 = np.mean(list(all_f1_scores.values()))
            print(f"Mean F1 Score: {mean_f1:.4f}")
            print(f"Total actions trained: {len(all_f1_scores)}")
        else:
            print("No models trained")

        return all_f1_scores

    def _train_section(self, section_df: pd.DataFrame, section_id: int) -> Dict[str, float]:

        single_data = {'features': [], 'labels': [], 'video_ids': []}
        pair_data = {'features': [], 'labels': [], 'video_ids': []}

        print("\n2. Processing videos and extracting features...")
        for idx, row in tqdm(section_df.iterrows(), total=len(section_df), desc="Videos"):
            video_info = self.loader.get_video_info(row)

            if len(video_info['behaviors_labeled']) == 0:
                continue

            try:
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
                    continue

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

                for (agent, target), actions in parsed_behaviors['single']:
                    mouse_id = int(agent.replace('mouse', ''))
                    features = self.feature_engineer.engineer_features(
                        pose_data, video_info, 'single', (mouse_id,)
                    )

                    if len(features) > 0:
                        key = (agent, target)
                        if key in labels_dict['single']:
                            single_data['features'].append(features)
                            single_data['labels'].append(labels_dict['single'][key])
                            single_data['video_ids'].append([video_info['video_id']] * len(features))

                for (agent, target), actions in parsed_behaviors['pair']:
                    agent_id = int(agent.replace('mouse', ''))
                    target_id = int(target.replace('mouse', ''))
                    features = self.feature_engineer.engineer_features(
                        pose_data, video_info, 'pair', (agent_id, target_id)
                    )

                    if len(features) > 0:
                        key = (agent, target)
                        if key in labels_dict['pair']:
                            pair_data['features'].append(features)
                            pair_data['labels'].append(labels_dict['pair'][key])
                            pair_data['video_ids'].append([video_info['video_id']] * len(features))

            except Exception as e:
                print(f"\n  Error processing {video_info['video_id']}: {e}")
                continue

        f1_scores = {}

        if len(single_data['features']) > 0:
            print("\n3. Training single behavior classifiers...")
            single_scores = self._train_behavior_type(single_data, section_id, 'single')
            f1_scores.update(single_scores)

        if len(pair_data['features']) > 0:
            print("\n4. Training pair behavior classifiers...")
            pair_scores = self._train_behavior_type(pair_data, section_id, 'pair')
            f1_scores.update(pair_scores)

        return f1_scores

    def _train_behavior_type(self, data: Dict, section_id: int, behavior_type: str) -> Dict[str, float]:

        X = pd.concat(data['features'], axis=0, ignore_index=True)
        y = pd.concat(data['labels'], axis=0, ignore_index=True)
        video_ids = pd.Series(np.concatenate(data['video_ids']))

        print(f"   Data shape: {X.shape}")
        print(f"   Actions: {list(y.columns)}")

        classifier = BehaviorClassifier(self.config)
        f1_scores = classifier.train_multiple_actions(
            X, y, video_ids, list(y.columns), f"{section_id}_{behavior_type}"
        )

        return f1_scores


class InferencePipeline:

    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.postprocessor = PredictionPostProcessor(config)

    def run(self, output_path: str = None) -> pd.DataFrame:
        print("=" * 60)
        print("INFERENCE PIPELINE")
        print("=" * 60)

        print("\n1. Loading test metadata...")
        test_df = self.loader.load_metadata(mode="test")
        print(f"   Loaded {len(test_df)} test videos")

        all_segments = []

        body_parts_groups = test_df.groupby('body_parts_tracked')

        for section_id, (body_parts_str, group_df) in enumerate(body_parts_groups):
            print(f"\n{'=' * 60}")
            print(f"Section {section_id + 1}/{len(body_parts_groups)}")
            print(f"{'=' * 60}")

            section_segments = self._predict_section(group_df, section_id)
            all_segments.extend(section_segments)

            gc.collect()

        submission = self.postprocessor.create_submission(all_segments, output_path)

        print(f"\n{'=' * 60}")
        print(f"Submission created: {len(submission)} segments")
        print(f"{'=' * 60}")

        return submission

    def _predict_section(self, section_df: pd.DataFrame, section_id: int) -> List[pd.DataFrame]:

        segments_list = []

        print("\n2. Processing videos and making predictions...")
        for idx, row in tqdm(section_df.iterrows(), total=len(section_df), desc="Videos"):
            video_info = self.loader.get_video_info(row)

            if len(video_info['behaviors_labeled']) == 0:
                continue

            try:
                tracking_df = self.loader.load_tracking_data(
                    video_info['lab_id'],
                    video_info['video_id'],
                    mode="test"
                )

                pose_data = self.processor.pivot_tracking_data(
                    tracking_df,
                    video_info['pix_per_cm']
                )

                parsed_behaviors = self.processor.parse_behaviors_labeled(
                    video_info['behaviors_labeled']
                )

                for (agent, target), actions in parsed_behaviors['single']:
                    segments = self._predict_agent_target(
                        pose_data, video_info, agent, target, actions,
                        section_id, 'single'
                    )
                    if len(segments) > 0:
                        segments_list.append(segments)

                for (agent, target), actions in parsed_behaviors['pair']:
                    segments = self._predict_agent_target(
                        pose_data, video_info, agent, target, actions,
                        section_id, 'pair'
                    )
                    if len(segments) > 0:
                        segments_list.append(segments)

            except Exception as e:
                print(f"\n  Error processing {video_info['video_id']}: {e}")
                continue

        return segments_list

    def _predict_agent_target(self, pose_data: pd.DataFrame, video_info: Dict,
                             agent: str, target: str, actions: List[str],
                             section_id: int, behavior_type: str) -> pd.DataFrame:

        if behavior_type == 'single':
            mouse_ids = (int(agent.replace('mouse', '')),)
        else:
            mouse_ids = (int(agent.replace('mouse', '')), int(target.replace('mouse', '')))

        features = self.feature_engineer.engineer_features(
            pose_data, video_info, behavior_type, mouse_ids
        )

        if len(features) == 0:
            return pd.DataFrame()

        classifier = BehaviorClassifier(self.config)
        classifier.load_multiple_models(actions, f"{section_id}_{behavior_type}")

        predictions = classifier.predict_multiple_actions(features, actions)

        metadata = pd.DataFrame({
            'video_id': video_info['video_id'],
            'agent_id': agent,
            'target_id': target,
            'video_frame': features.index
        })

        segments = self.postprocessor.predictions_to_segments(
            predictions, metadata, classifier.thresholds
        )

        return segments
