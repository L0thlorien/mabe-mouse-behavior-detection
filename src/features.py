import numpy as np
import pandas as pd
import itertools
from typing import List


class FeatureExtractor:

    def __init__(self, config):
        self.config = config

    def extract_features(self, pose_data: pd.DataFrame, video_info: dict,
                        behavior_type: str = 'single',
                        mouse_ids: tuple = None) -> pd.DataFrame:
        fps = video_info['fps']
        body_parts = video_info['body_parts_tracked']

        if behavior_type == 'single':
            return self._extract_single_features(pose_data, mouse_ids[0], body_parts, fps)
        else:
            return self._extract_pair_features(pose_data, mouse_ids[0], mouse_ids[1], body_parts, fps)

    def _extract_single_features(self, pose_data: pd.DataFrame, mouse_id: int,
                                 body_parts: List[str], fps: float) -> pd.DataFrame:
        features = pd.DataFrame(index=pose_data.index)

        try:
            mouse_data = pose_data[mouse_id]
        except KeyError:
            return features

        available_parts = [p for p in body_parts if p in mouse_data.columns]

        for p1, p2 in itertools.combinations(available_parts, 2):
            if p1 in mouse_data.columns and p2 in mouse_data.columns:
                dist_sq = ((mouse_data[p1]['x'] - mouse_data[p2]['x'])**2 +
                          (mouse_data[p1]['y'] - mouse_data[p2]['y'])**2)
                features[f'{p1}_{p2}_dist'] = np.sqrt(dist_sq)

        if 'body_center' in available_parts:
            center_x = mouse_data['body_center']['x']
            center_y = mouse_data['body_center']['y']

            speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * fps
            features['speed'] = speed

            for window in self.config.TEMPORAL_WINDOWS:
                w = self._normalize_window(window, fps)
                features[f'speed_mean_{window}'] = speed.rolling(w, min_periods=1, center=True).mean()
                features[f'speed_std_{window}'] = speed.rolling(w, min_periods=1, center=True).std()
                features[f'x_std_{window}'] = center_x.rolling(w, min_periods=1, center=True).std()
                features[f'y_std_{window}'] = center_y.rolling(w, min_periods=1, center=True).std()

        if 'nose' in available_parts and 'tail_base' in available_parts:
            nose_tail_dist = np.sqrt(
                (mouse_data['nose']['x'] - mouse_data['tail_base']['x'])**2 +
                (mouse_data['nose']['y'] - mouse_data['tail_base']['y'])**2
            )
            features['nose_tail_dist'] = nose_tail_dist

            if 'ear_left' in available_parts and 'ear_right' in available_parts:
                ear_dist = np.sqrt(
                    (mouse_data['ear_left']['x'] - mouse_data['ear_right']['x'])**2 +
                    (mouse_data['ear_left']['y'] - mouse_data['ear_right']['y'])**2
                )
                features['elongation'] = nose_tail_dist / (ear_dist + 1e-6)

        features = features.fillna(0)

        return features

    def _extract_pair_features(self, pose_data: pd.DataFrame,
                              agent_id: int, target_id: int,
                              body_parts: List[str], fps: float) -> pd.DataFrame:
        features = pd.DataFrame(index=pose_data.index)

        try:
            agent_data = pose_data[agent_id]
            target_data = pose_data[target_id]
        except KeyError:
            return features

        available_parts_agent = [p for p in body_parts if p in agent_data.columns]
        available_parts_target = [p for p in body_parts if p in target_data.columns]

        for p_agent in available_parts_agent:
            for p_target in available_parts_target:
                dist_sq = ((agent_data[p_agent]['x'] - target_data[p_target]['x'])**2 +
                          (agent_data[p_agent]['y'] - target_data[p_target]['y'])**2)
                features[f'agent_{p_agent}_target_{p_target}_dist'] = np.sqrt(dist_sq)

        if 'body_center' in available_parts_agent and 'body_center' in available_parts_target:
            center_dist = np.sqrt(
                (agent_data['body_center']['x'] - target_data['body_center']['x'])**2 +
                (agent_data['body_center']['y'] - target_data['body_center']['y'])**2
            )
            features['center_dist'] = center_dist

            approach = -center_dist.diff() * fps
            features['approach_rate'] = approach

            for window in self.config.TEMPORAL_WINDOWS:
                w = self._normalize_window(window, fps)
                features[f'center_dist_mean_{window}'] = center_dist.rolling(w, min_periods=1, center=True).mean()
                features[f'center_dist_std_{window}'] = center_dist.rolling(w, min_periods=1, center=True).std()
                features[f'approach_mean_{window}'] = approach.rolling(w, min_periods=1, center=True).mean()

            features['very_close'] = (center_dist < 5.0).astype(float)
            features['close'] = ((center_dist >= 5.0) & (center_dist < 15.0)).astype(float)
            features['medium'] = ((center_dist >= 15.0) & (center_dist < 30.0)).astype(float)
            features['far'] = (center_dist >= 30.0).astype(float)

        if 'body_center' in available_parts_agent and 'body_center' in available_parts_target:
            agent_vx = agent_data['body_center']['x'].diff() * fps
            agent_vy = agent_data['body_center']['y'].diff() * fps
            target_vx = target_data['body_center']['x'].diff() * fps
            target_vy = target_data['body_center']['y'].diff() * fps

            agent_speed = np.sqrt(agent_vx**2 + agent_vy**2)
            target_speed = np.sqrt(target_vx**2 + target_vy**2)

            features['agent_speed'] = agent_speed
            features['target_speed'] = target_speed

            features['speed_diff'] = agent_speed - target_speed

            vel_alignment = agent_vx * target_vx + agent_vy * target_vy
            features['vel_alignment'] = vel_alignment

        if 'nose' in available_parts_agent and 'nose' in available_parts_target:
            nose_nose_dist = np.sqrt(
                (agent_data['nose']['x'] - target_data['nose']['x'])**2 +
                (agent_data['nose']['y'] - target_data['nose']['y'])**2
            )
            features['nose_nose_dist'] = nose_nose_dist

            features['noses_close'] = (nose_nose_dist < 3.0).astype(float)

        features = features.fillna(0)

        return features

    def _normalize_window(self, window: int, fps: float) -> int:
        return max(1, int(round(window * fps / self.config.FPS_REFERENCE)))


class FeatureEngineer:

    def __init__(self, config):
        self.config = config
        self.extractor = FeatureExtractor(config)

    def engineer_features(self, pose_data: pd.DataFrame, video_info: dict,
                         behavior_type: str, mouse_ids: tuple) -> pd.DataFrame:
        features = self.extractor.extract_features(pose_data, video_info, behavior_type, mouse_ids)


        return features

    def _add_custom_features(self, features: pd.DataFrame,
                            pose_data: pd.DataFrame,
                            video_info: dict) -> pd.DataFrame:
        return features
