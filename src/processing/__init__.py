import pandas as pd

from .find_intervals import find_annotation_intervals


def batch_intervals():
    df = pd.read_csv("metadata/annotations.csv")
    intervals_df = find_annotation_intervals(df)
    print(intervals_df)

# download_segment("P01", "P01_104", "00:00:05", "00:00:10", "test.mp4")