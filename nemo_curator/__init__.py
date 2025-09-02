import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from cosmos_xenna.ray_utils.cluster import API_LIMIT

# We set these incase a user ever starts a ray cluster with nemo_curator, we need these for Xenna to work
os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = str(API_LIMIT)
os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = str(API_LIMIT)
