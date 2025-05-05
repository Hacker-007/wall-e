from typing import List
import os
import pandas as pd

from .find_intervals import find_annotation_intervals

def get_intervals() -> pd.DataFrame:
    interval_path = "metadata/intervals.csv"
    if os.path.exists(interval_path):
        return pd.read_csv(interval_path)

    annotation_df = pd.read_csv("metadata/annotations.csv")
    intervals_df = find_annotation_intervals(annotation_df)
    sampled_df = intervals_df.groupby("middle_noun", group_keys=False).apply(
        lambda rows: rows.sample(n=1000, random_state=42)
    )

    sampled_df.to_csv(interval_path, index=False)
    return sampled_df


def batch_intervals(num_groups: int) -> List[pd.DataFrame]:
    intervals_df = get_intervals()
    num_intervals = len(intervals_df)
    batches = []
    batch_size = num_intervals // num_groups
    for start_row in range(0, num_intervals, batch_size):
        end_row = min(start_row + batch_size, num_intervals)
        batch_df = intervals_df.iloc[start_row:end_row]
        batch_df = batch_df.sort_values(by=['participant_id', 'video_id'])
        batches.append(batch_df)

    return batches

# download_segment("P01", "P01_104", "00:00:05", "00:00:10", "test.mp4")
