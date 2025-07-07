import pathlib
from ray_curator.utils.storage_client import StoragePrefix


def _get_local_path(localpath: pathlib.Path, *args: str) -> pathlib.Path:
    """Construct a full local path from a base path and additional components.

    Args:
        localpath: The base local path.
        *args: Additional path components.

    Returns:
        The full local path as a Path object.

    """
    return pathlib.Path(localpath, *args)


def get_full_path(path: str | StoragePrefix | pathlib.Path, *args: str) -> StoragePrefix | pathlib.Path:
    """Construct a full path from a base path and additional components.

    Args:
        path: The base path.
        *args: Additional path components.

    Returns:
        The full path as a StoragePrefix or Path object.

    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        return _get_local_path(pathlib.Path(path), *args)

    assert isinstance(path, pathlib.Path), "only str or pathlib.Path paths are currently supported."
    return pathlib.Path(path, *args)

