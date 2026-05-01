#!/usr/bin/env python3
"""Verify Ray cluster can initialize and stop."""
from nemo_curator.core.client import RayClient

rc = RayClient()
rc.start()
try:
    print("✓ Ray cluster initialized successfully")
finally:
    rc.stop()
    print("✓ Ray cluster stopped")
