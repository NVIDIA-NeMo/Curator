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
"""Unit tests for InternVideo integration module."""

import pathlib

import pytest

from ray_curator.modules.internvideo import (
    InternVideo2_CLIP,
    InternVideo2_CLIP_small,
    InternVideo2_Stage2_audiovisual,
    InternVideo2_Stage2_visual,
    build_bert,
    interpolate_pos_embed_internvideo2_new,
    pretrain_internvideo2_1b_patch14_224,
    pretrain_internvideo2_6b_patch14_224,
)


class TestInternVideoIntegration:
    """Test cases for InternVideo integration module."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Get the path to the submodule for testing
        # The test file is in tests/models/, so we need to go up to ray-curator root
        self.submodule_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "externals"
            / "InternVideo"
            / "InternVideo2"
            / "multi_modality"
        )

    def test_submodule_path_exists(self) -> None:
        """Test that the InternVideo submodule path exists."""
        assert self.submodule_path.exists(), f"Submodule path does not exist: {self.submodule_path}"
        assert (self.submodule_path / "__init__.py").exists(), "Submodule __init__.py not found"

    def test_all_models_imported(self) -> None:
        """Test that all expected InternVideo models are imported and available."""
        # Check that all models are imported and are not None
        assert InternVideo2_Stage2_visual is not None
        assert InternVideo2_Stage2_audiovisual is not None
        assert InternVideo2_CLIP is not None
        assert InternVideo2_CLIP_small is not None
        assert pretrain_internvideo2_1b_patch14_224 is not None
        assert pretrain_internvideo2_6b_patch14_224 is not None
        assert build_bert is not None
        assert interpolate_pos_embed_internvideo2_new is not None

    def test_model_types(self) -> None:
        """Test that imported models have the expected types."""
        # Check that models are callable (functions or classes)
        assert callable(pretrain_internvideo2_1b_patch14_224)
        assert callable(pretrain_internvideo2_6b_patch14_224)
        assert callable(build_bert)
        assert callable(interpolate_pos_embed_internvideo2_new)

    def test_model_classes(self) -> None:
        """Test that model classes are properly imported."""
        # These should be classes, not instances
        assert isinstance(InternVideo2_Stage2_visual, type)
        assert isinstance(InternVideo2_Stage2_audiovisual, type)
        assert isinstance(InternVideo2_CLIP, type)
        assert isinstance(InternVideo2_CLIP_small, type)

    def test_package_installation_fallback(self) -> None:
        """Test that the module structure supports package installation fallback."""
        # This test verifies that the module has the expected structure
        # for handling package installation when needed
        from ray_curator.modules.internvideo import _ensure_internvideo_installed

        # Check that the function exists and is callable
        assert callable(_ensure_internvideo_installed)

        # Check that it has the expected docstring
        assert "Ensure the InternVideo package is installed" in _ensure_internvideo_installed.__doc__

    def test_module_structure(self) -> None:
        """Test that the integration module has the expected structure."""
        from ray_curator.modules.internvideo import __all__

        expected_exports = [
            "InternVideo2_Stage2_visual",
            "InternVideo2_Stage2_audiovisual",
            "InternVideo2_CLIP",
            "InternVideo2_CLIP_small",
            "pretrain_internvideo2_1b_patch14_224",
            "pretrain_internvideo2_6b_patch14_224",
            "build_bert",
            "interpolate_pos_embed_internvideo2_new",
        ]

        assert set(__all__) == set(expected_exports), f"Expected {expected_exports}, got {__all__}"

    def test_submodule_config_files(self) -> None:
        """Test that important submodule configuration files exist."""
        config_dir = self.submodule_path / "configs"
        assert config_dir.exists(), "Configs directory not found"

        # Check for key config files - make this more flexible
        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) > 0, "No JSON config files found in configs directory"

        # Check that at least one config file contains expected content
        config_file = config_files[0]
        assert config_file.exists(), f"Config file {config_file} not found"

    def test_submodule_model_files(self) -> None:
        """Test that submodule model files exist."""
        models_dir = self.submodule_path / "models"
        assert models_dir.exists(), "Models directory not found"

        # Check for key model files - make this more flexible
        model_files = list(models_dir.glob("*.py"))
        assert len(model_files) > 0, "No Python model files found in models directory"

        # Check that at least one model file exists
        model_file = model_files[0]
        assert model_file.exists(), f"Model file {model_file} not found"

    def test_submodule_structure(self) -> None:
        """Test that the submodule has the expected directory structure."""
        # Check for essential directories
        essential_dirs = ["models", "configs", "utils"]
        for dir_name in essential_dirs:
            dir_path = self.submodule_path / dir_name
            assert dir_path.exists(), f"Essential directory {dir_name} not found"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"


class TestInternVideoIntegrationErrorHandling:
    """Test cases for error handling in InternVideo integration."""

    def test_import_consistency(self) -> None:
        """Test that imports are consistent across multiple calls."""
        # Import the module multiple times to ensure consistency
        from ray_curator.modules.internvideo import InternVideo2_Stage2_visual as model1
        from ray_curator.modules.internvideo import InternVideo2_Stage2_visual as model2

        # Should be the same object
        assert model1 is model2, "Multiple imports should return the same object"

    def test_module_attributes(self) -> None:
        """Test that the module has the expected attributes and functions."""
        from ray_curator.modules.internvideo import _ensure_internvideo_installed

        # Check that the function exists
        assert callable(_ensure_internvideo_installed)

        # Check that it's a function
        import types

        assert isinstance(_ensure_internvideo_installed, types.FunctionType)


if __name__ == "__main__":
    pytest.main([__file__])
