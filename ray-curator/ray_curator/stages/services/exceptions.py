# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for handling exceptions in Ray-distributed environments.
"""

import openai


def make_ray_serializable_exception(e: Exception) -> Exception:
    """
    Convert OpenAI exceptions to Ray-serializable exceptions while preserving
    important error information for retry logic.

    The main issue is that OpenAI SDK v1.30+ changed APIStatusError constructor
    signature, making it incompatible with Ray's pickle serialization.
    We only convert APIStatusError and preserve other exceptions as-is.

    Args:
        e: The original exception

    Returns:
        A Ray-serializable exception that preserves error information
    """
    if isinstance(e, openai.APIStatusError):
        # Preserve the exact status code and message for retry logic
        # This ensures "429" and "500" are still detectable in error strings
        status_code = getattr(e, "status_code", "unknown")
        message = getattr(e, "message", str(e))

        # Create a message that preserves the original error format
        # so retry logic can still detect "429" for rate limits
        error_msg = f"Error code: {status_code} - {message}"
        return RuntimeError(error_msg)
    else:
        # For all other exceptions (including RateLimitError, etc.),
        # preserve them as-is since they don't have Ray serialization issues
        return e
