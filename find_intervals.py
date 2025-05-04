import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def timestamp_to_seconds(t_str):
    t = datetime.strptime(t_str, '%H:%M:%S.%f')
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
    return delta.total_seconds()

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
        ends   = group["stop_seconds"].values
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

if __name__ == '__main__':
    kept = find_single_action_intervals_from_csv('EPIC_100_train.csv')
    print(kept[0])