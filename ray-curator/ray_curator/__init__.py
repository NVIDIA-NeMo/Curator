import os

from cosmos_xenna.ray_utils.cluster import API_LIMIT

# We set these incase a user ever starts a ray cluster with ray_curator, we need these for Xenna to work
os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = str(API_LIMIT)
os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = str(API_LIMIT)
