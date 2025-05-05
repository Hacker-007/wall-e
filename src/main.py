from processing import batch_intervals

batches = batch_intervals(2)
for batch in batches:
    print(batch)
