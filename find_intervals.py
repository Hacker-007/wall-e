import pandas as pd
import numpy as np
from datetime import datetime


def timestamp_to_seconds(t):
    """Convert HH:MM:SS.FF to seconds"""
    t = datetime.strptime(t, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def find_single_action_intervals_from_csv(df, bin_size=1.0):
    """
    Reads a CSV with columns [video_id, start_seconds, stop_seconds, action_noun]
    and returns a dict mapping each video_id to a list of (bin_start, bin_end, action_label)
    for 1s bins where exactly one action overlaps.
    """
    kept = [[] for _ in range(len(df))]

    # group annotations by video
    for vid, group in df.groupby("video_id"):
        # get numpy arrays for fast filtering
        starts = group["start_seconds"].values
        ends = group["stop_seconds"].values
        idxs = group.index.values

        max_t = ends.max()
        edges = np.arange(0, np.ceil(max_t) + bin_size, bin_size)

        # iterate over bins
        for t0, t1 in zip(edges[:-1], edges[1:]):
            # boolean mask of actions overlapping [t0, t1)
            overlap_mask = ~((ends <= t0) | (starts >= t1))
            if overlap_mask.sum() == 1:
                idx = idxs[overlap_mask][0]
                kept[idx].append((t0, t1))

        # single_action_bins[vid] = kept
    return kept


def find_free_1s_intervals(df, step=1.0):
    df = df.copy()
    df["start_sec"] = df["start_timestamp"].apply(timestamp_to_seconds)
    df["stop_sec"] = df["stop_timestamp"].apply(timestamp_to_seconds)

    result = []

    for video_id, group in df.groupby("video_id"):
        min_time = group["start_sec"].min()
        max_time = group["stop_sec"].max()

        t = min_time
        while t + 5.0 <= max_time:
            window_start = t
            window_end = t + 5.0
            middle_start = t + 2.0
            middle_end = t + 3.0

            # Find all annotations that overlap with the middle 1-second interval
            overlaps = group[
                (group["start_sec"] < middle_end) & (group["stop_sec"] > middle_start)
            ]

            if len(overlaps) == 1:
                verb = overlaps.iloc[0]["verb"]
                result.append(
                    {
                        "video_id": video_id,
                        "window_start": round(window_start, 2),
                        "window_end": round(window_end, 2),
                        "middle_start": round(middle_start, 2),
                        "middle_end": round(middle_end, 2),
                        "verb": verb,
                    }
                )

            t += step  # slide window forward

    return pd.DataFrame(result)


if __name__ == "__main__":
    kept = find_single_action_intervals_from_csv("EPIC_100_train.csv")
    print(kept[0])

    df = pd.read_csv("metadata/annotations.csv")
    free_df = find_free_1s_intervals(df)
    print(free_df)
