#!/usr/bin/env python3
"""Verify Ray cluster can initialize and stop."""
from nemo_curator.core.client import RayClient

rc = RayClient()
rc.start()
print("✓ Ray cluster initialized successfully")
rc.stop()
print("✓ Ray cluster stopped")
