import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.ndimage import median_filter


class PredictionPostProcessor:

    def __init__(self, config):
        self.config = config

    def predictions_to_segments(self, predictions: pd.DataFrame,
                               metadata: pd.DataFrame,
                               thresholds: Dict[str, float]) -> pd.DataFrame:
        segments = []

        for action in predictions.columns:
            threshold = thresholds.get(action, 0.5)

            binary_pred = (predictions[action] >= threshold).astype(int)

            if self.config.SMOOTH_PREDICTIONS and len(binary_pred) > self.config.SMOOTH_WINDOW:
                binary_pred = median_filter(binary_pred, size=self.config.SMOOTH_WINDOW)

            binary_array = binary_pred.values if hasattr(binary_pred, 'values') else binary_pred

            segments_action = self._find_segments(
                binary_array,
                metadata,
                action
            )

            segments.extend(segments_action)

        if len(segments) == 0:
            return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action',
                                        'start_frame', 'stop_frame'])

        df_segments = pd.DataFrame(segments)

        df_segments = self._filter_short_segments(df_segments)

        if self.config.MERGE_GAP_THRESHOLD > 0:
            df_segments = self._merge_nearby_segments(df_segments)

        return df_segments

    def _find_segments(self, binary_pred: np.ndarray, metadata: pd.DataFrame,
                      action: str) -> List[Dict]:
        segments = []

        changes = np.where(np.diff(binary_pred) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(binary_pred)]])

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            if binary_pred[start_idx] == 1:
                start_frame = metadata.iloc[start_idx]['video_frame']
                end_frame = metadata.iloc[end_idx - 1]['video_frame'] + 1

                segments.append({
                    'video_id': metadata.iloc[start_idx]['video_id'],
                    'agent_id': metadata.iloc[start_idx]['agent_id'],
                    'target_id': metadata.iloc[start_idx]['target_id'],
                    'action': action,
                    'start_frame': start_frame,
                    'stop_frame': end_frame
                })

        return segments

    def _filter_short_segments(self, segments: pd.DataFrame) -> pd.DataFrame:
        if len(segments) == 0:
            return segments

        duration = segments['stop_frame'] - segments['start_frame']
        valid_mask = duration >= self.config.MIN_BEHAVIOR_DURATION

        return segments[valid_mask].reset_index(drop=True)

    def _merge_nearby_segments(self, segments: pd.DataFrame) -> pd.DataFrame:
        if len(segments) == 0:
            return segments

        merged = []

        for group_key, group in segments.groupby(['video_id', 'agent_id', 'target_id', 'action']):
            group = group.sort_values('start_frame')

            current_start = None
            current_stop = None

            for _, row in group.iterrows():
                if current_start is None:
                    current_start = row['start_frame']
                    current_stop = row['stop_frame']
                elif row['start_frame'] - current_stop <= self.config.MERGE_GAP_THRESHOLD:
                    current_stop = row['stop_frame']
                else:
                    merged.append({
                        'video_id': group_key[0],
                        'agent_id': group_key[1],
                        'target_id': group_key[2],
                        'action': group_key[3],
                        'start_frame': current_start,
                        'stop_frame': current_stop
                    })
                    current_start = row['start_frame']
                    current_stop = row['stop_frame']

            if current_start is not None:
                merged.append({
                    'video_id': group_key[0],
                    'agent_id': group_key[1],
                    'target_id': group_key[2],
                    'action': group_key[3],
                    'start_frame': current_start,
                    'stop_frame': current_stop
                })

        return pd.DataFrame(merged)

    def create_submission(self, all_segments: List[pd.DataFrame],
                         output_path: str = None) -> pd.DataFrame:
        if len(all_segments) == 0:
            submission = pd.DataFrame({
                'video_id': [0],
                'agent_id': ['mouse1'],
                'target_id': ['self'],
                'action': ['rear'],
                'start_frame': [0],
                'stop_frame': [100]
            })
        else:
            submission = pd.concat(all_segments, ignore_index=True)

        submission = submission.sort_values(['video_id', 'agent_id', 'target_id',
                                            'action', 'start_frame'])
        submission = submission.reset_index(drop=True)
        submission.index.name = 'row_id'

        if output_path:
            submission.to_csv(output_path)
            print(f"Submission saved to {output_path}")

        return submission


class SubmissionValidator:

    @staticmethod
    def validate(submission: pd.DataFrame) -> bool:
        required_columns = ['video_id', 'agent_id', 'target_id', 'action',
                          'start_frame', 'stop_frame']

        for col in required_columns:
            if col not in submission.columns:
                print(f"Missing column: {col}")
                return False

        invalid = submission['start_frame'] >= submission['stop_frame']
        if invalid.any():
            print(f"Found {invalid.sum()} rows with start_frame >= stop_frame")
            return False

        for key, group in submission.groupby(['video_id', 'agent_id', 'target_id', 'action']):
            group = group.sort_values('start_frame')
            for i in range(len(group) - 1):
                if group.iloc[i]['stop_frame'] > group.iloc[i + 1]['start_frame']:
                    print(f"Found overlapping segments for {key}")
                    return False

        print("Submission validation passed!")
        return True
