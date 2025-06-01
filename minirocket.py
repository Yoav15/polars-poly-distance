import numpy as np
import polars as pl
from ppd import mini_rocket_expr
import time
import matplotlib.pyplot as plt

# Parameters
num_kernels = 10_000  # Fixed
series_counts = [10, 100, 500, 1000]  # Vary this
series_length = 100  # Fixed
weights_choices = [-1, 2]

# Benchmark results
results = []

for num_series in series_counts:
    print(f"\nRunning with {num_series} series, {num_kernels} kernels, series length {series_length}")

    # Generate synthetic time series data
    time_series_data = [np.random.randn(series_length).astype(np.float32) for _ in range(num_series)]

    # Generate kernels
    kernels = []
    for _ in range(num_kernels):
        while True:
            kernel = np.random.choice(weights_choices, size=9)  # Kernel length 9
            if kernel.sum() == 0:
                break
        kernels.append(kernel.astype(np.float32))

    # Generate dilations
    dilations = np.random.randint(1, max(2, series_length // 9), size=num_kernels).astype(np.float32)

    # Generate biases
    biases = np.random.uniform(-1, 1, size=num_kernels).astype(np.float32)

    # Create Polars DataFrame
    time_series_data_df = pl.DataFrame({
        "tracks": [pl.Series(ts) for ts in time_series_data]
    })

    # Convert kernels to Polars Series
    kernels_col = pl.Series([pl.Series(k) for k in kernels])
    dilations_col = pl.Series(dilations)
    biases_col = pl.Series(biases)

    # Benchmark
    start = time.time()
    output = time_series_data_df.with_columns(
        mini_rocket_expr(pl.col('tracks'), kernels_col, dilations_col, biases_col).alias('res')
    )
    end = time.time()
    elapsed = end - start
    print(f"Time: {elapsed:.2f} sec")

    results.append((num_series, elapsed))

# Plot
plt.figure(figsize=(8, 5))
x_vals, y_vals = zip(*results)
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
plt.xlabel("# of Tracks")
plt.ylabel("Execution Time (seconds)")
plt.title(f"MiniROCKET Rust Plugin Benchmark\n(num_kernels={num_kernels}, series_length={series_length})")
plt.grid(True)
plt.tight_layout()
plt.show()
