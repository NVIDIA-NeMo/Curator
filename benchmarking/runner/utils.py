from typing import Any

def get_obj_for_json(obj: Any) -> Any:
    """
    Recursively convert objects to Python primitives for JSON serialization. 
    Useful for objects like Path, sets, bytes, etc.
    """
    if isinstance(obj, dict):
        return {get_obj_for_json(k): get_obj_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [get_obj_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return [get_obj_for_json(item) for item in obj]
    elif hasattr(obj, "as_posix"):  # Path objects
        return obj.as_posix()
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif hasattr(obj, "to_json") and callable(getattr(obj, "to_json")):
        return obj.to_json()
    elif hasattr(obj, "__dict__"):
        return get_obj_for_json(vars(obj))
    elif obj is None:
        return "null"
    elif isinstance(obj, str) and len(obj) == 0:  # special case for Slack, empty strings not allowed
        return " "
    else:
        return obj
