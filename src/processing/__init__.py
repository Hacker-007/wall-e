import os
import pandas as pd

from .find_intervals import find_annotation_intervals


def get_intervals() -> pd.DataFrame:
    interval_path = "metadata/intervals.csv"
    if os.path.exists(interval_path):
        return pd.read_csv(interval_path)

    annotation_df = pd.read_csv("metadata/annotations.csv")
    interval_df = find_annotation_intervals(annotation_df)
    interval_df.to_csv(interval_path)
    return interval_df


def batch_intervals():
    intervals_df = get_intervals()
    print(intervals_df)


# download_segment("P01", "P01_104", "00:00:05", "00:00:10", "test.mp4")
