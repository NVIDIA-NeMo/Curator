import pytest

from ray_curator.utils.column_utils import resolve_filename_column


class TestResolveFilenameColumn:
    """Test cases for resolve_filename_column function."""

    def test_resolve_filename_column_true(self):
        """Test that True returns the default filename column name."""
        assert resolve_filename_column(True) == "file_name"

        assert resolve_filename_column(False) is None

        custom_name = "custom_filename"
        assert resolve_filename_column(custom_name) == custom_name

        assert resolve_filename_column("") == ""

        with pytest.raises(ValueError, match="Invalid value for add_filename_column"):
            resolve_filename_column(None)  # type: ignore[reportArgumentType]
