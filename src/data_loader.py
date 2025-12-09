import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import gc


class DataLoader:

    def __init__(self, config):
        self.config = config

    def load_metadata(self, mode: str = "train") -> pd.DataFrame:
        if mode == "train":
            df = pd.read_csv(self.config.TRAIN_CSV)
            df['n_mice'] = 4 - df[['mouse1_strain', 'mouse2_strain',
                                   'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

            if self.config.EXCLUDE_MABE22:
                df = df[~df['lab_id'].str.startswith('MABe22')].reset_index(drop=True)

        else:
            df = pd.read_csv(self.config.TEST_CSV)

        return df

    def load_tracking_data(self, lab_id: str, video_id: str, mode: str = "train") -> pd.DataFrame:
        if mode == "train":
            path = self.config.TRAIN_TRACKING_DIR / lab_id / f"{video_id}.parquet"
        else:
            path = self.config.TEST_TRACKING_DIR / lab_id / f"{video_id}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Tracking file not found: {path}")

        return pd.read_parquet(path)

    def load_annotation_data(self, lab_id: str, video_id: str) -> Optional[pd.DataFrame]:
        path = self.config.TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"

        if not path.exists():
            return None

        return pd.read_parquet(path)

    def get_video_info(self, row: pd.Series) -> Dict:
        behaviors_labeled = []
        if isinstance(row.get('behaviors_labeled'), str):
            behaviors_labeled = json.loads(row['behaviors_labeled'])
            behaviors_labeled = sorted(list({b.replace("'", "") for b in behaviors_labeled}))

        return {
            'lab_id': row['lab_id'],
            'video_id': row['video_id'],
            'fps': row.get('frames_per_second', 30.0),
            'pix_per_cm': row.get('pix_per_cm_approx', 12.0),
            'n_mice': row.get('n_mice', 3),
            'behaviors_labeled': behaviors_labeled,
            'body_parts_tracked': json.loads(row['body_parts_tracked'])
        }


class DataProcessor:

    def __init__(self, config):
        self.config = config

    def pivot_tracking_data(self, tracking_df: pd.DataFrame, pix_per_cm: float) -> pd.DataFrame:
        pivoted = tracking_df.pivot(
            columns=['mouse_id', 'bodypart'],
            index='video_frame',
            values=['x', 'y']
        )

        pivoted = pivoted.reorder_levels([1, 2, 0], axis=1).T.sort_index().T

        pivoted = pivoted / pix_per_cm

        return pivoted

    def create_labels_for_video(self, annotation_df: pd.DataFrame,
                                  video_info: Dict,
                                  n_frames: int) -> Dict[str, pd.DataFrame]:
        labels = {'single': {}, 'pair': {}}

        for _, ann in annotation_df.iterrows():
            agent = f"mouse{ann['agent_id']}"
            target = f"mouse{ann['target_id']}" if ann['target_id'] != ann['agent_id'] else 'self'
            action = ann['action']
            start, stop = ann['start_frame'], ann['stop_frame']

            behavior_type = 'single' if target == 'self' else 'pair'

            key = (agent, target)
            if key not in labels[behavior_type]:
                labels[behavior_type][key] = pd.DataFrame(
                    0, index=range(n_frames), columns=[]
                )

            if action not in labels[behavior_type][key].columns:
                labels[behavior_type][key][action] = 0

            labels[behavior_type][key].loc[start:stop, action] = 1

        return labels

    def parse_behaviors_labeled(self, behaviors_labeled: List[str]) -> Dict[str, List[Tuple]]:
        parsed = {'single': [], 'pair': []}

        for behavior_str in behaviors_labeled:
            parts = behavior_str.split(',')
            if len(parts) == 3:
                agent, target, action = parts
                behavior_type = 'single' if target == 'self' else 'pair'

                key = (agent, target)
                existing = [item for item in parsed[behavior_type] if item[0] == key]

                if existing:
                    existing[0][1].append(action)
                else:
                    parsed[behavior_type].append((key, [action]))

        return parsed

    def normalize_fps(self, window_size: int, fps: float) -> int:
        return max(1, int(round(window_size * fps / self.config.FPS_REFERENCE)))


def get_fps(video_info: Dict, default: float = 30.0) -> float:
    return float(video_info.get('fps', default))
