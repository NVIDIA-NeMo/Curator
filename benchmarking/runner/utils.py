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


def get_obj_for_json(obj: object) -> str | int | float | bool | list | dict:
    """
    Recursively convert objects to Python primitives for JSON serialization.
    Useful for objects like Path, sets, bytes, etc.
    """
    if isinstance(obj, dict):
        retval = {get_obj_for_json(k): get_obj_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [get_obj_for_json(item) for item in obj]
    elif hasattr(obj, "as_posix"):  # Path objects
        retval = obj.as_posix()
    elif isinstance(obj, bytes):
        retval = obj.decode("utf-8", errors="replace")
    elif hasattr(obj, "to_json") and callable(obj.to_json):
        retval = obj.to_json()
    elif hasattr(obj, "__dict__"):
        retval = get_obj_for_json(vars(obj))
    elif obj is None:
        retval = "null"
    elif isinstance(obj, str) and len(obj) == 0:  # special case for Slack, empty strings not allowed
        retval = " "
    else:
        retval = obj
    return retval
