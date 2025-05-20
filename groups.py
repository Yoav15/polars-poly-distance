import polars as pl
import numpy as np
import time

# effects polars impl more
num_points = 5
# effects naive impl more
group_count = 100

df = pl.DataFrame(
    {
        "group_id": np.concatenate([[i]*num_points for i in range(group_count)]),
        "data": np.concatenate([np.random.rand(num_points) for _ in range(group_count)]), 
    }
)

def distance_between_groups(df: pl.DataFrame):
    """
    for each pair of groups g1, g2, returns some aggregate function of their data, 
    such that we get a G x G matrix of the resulting aggregations.
    """
    groups = df['group_id'].unique(maintain_order=True)
    res = np.zeros((df['group_id'].n_unique(), df['group_id'].n_unique()))
    for i, g1 in enumerate(groups):
        g1_df = df.filter(pl.col('group_id') == g1)
        for j, g2 in enumerate(groups):
            g2_df = df.filter(pl.col('group_id') == g2)
            res[i][j] = (g1_df['data'] * g2_df['data']).sum()

    return res


def polars_distance(df: pl.DataFrame):
    df = df.with_columns(index=pl.int_range(pl.len()).over("group_id"))

    res = (
        df.join(df, on="index")
        # group by the pair of groups
        # maintain_order to keep the same order as the numpy array (optional)
        .group_by("group_id", "group_id_right", maintain_order=True)
        .agg(pl.col("data").dot("data_right"))
    )
    return res

t0 = time.perf_counter()
res = distance_between_groups(df)
print(res)
t1 = time.perf_counter()
print(t1 - t0)


t0 = time.perf_counter()
res = polars_distance(df)
print(res)
t1 = time.perf_counter()
print(t1 - t0)