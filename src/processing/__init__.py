import os
import pandas as pd
from tqdm import tqdm

from .find_intervals import find_annotation_intervals
from .data_fetching import get_data_url

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


def batch_intervals():
    intervals_df = get_intervals()
    print(intervals_df)

# download_segment("P01", "P01_104", "00:00:05", "00:00:10", "test.mp4")
