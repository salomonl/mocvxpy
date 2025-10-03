from dask.distributed import Client

# Create global client to avoid warnings for all parallel
# tests
CLIENT = Client(n_workers=2, threads_per_worker=2)
