"""Unit tests for pos_embed.py position embedding functions."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from ray_curator.models.pos_embed import (
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    get_3d_sincos_pos_embed,
    interpolate_pos_embed,
    interpolate_pos_embed_internvideo2,
    interpolate_pos_embed_internvideo2_new,
)


class TestPositionEmbeddingFunctions:
    """Test cases for position embedding functions."""

    def test_get_1d_sincos_pos_embed_from_grid(self) -> None:
        """Test get_1d_sincos_pos_embed_from_grid function."""
        embed_dim = 8
        pos = np.array([0, 1, 2, 3])

        result = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

        assert result.shape == (4, 8)
        assert result.dtype == np.float64

        # First half should be sin values, second half should be cos values
        # For position 0, sin should be 0 and cos should be 1
        assert np.allclose(result[0, :4], 0, atol=1e-10)  # sin(0) = 0
        assert np.allclose(result[0, 4:], 1, atol=1e-10)  # cos(0) = 1

        # Check that other positions have non-zero values
        assert not np.allclose(result[1:, :4], 0, atol=1e-10)  # sin values should not all be 0
        assert not np.allclose(result[1:, 4:], 0, atol=1e-10)  # cos values should not all be 0

    def test_get_1d_sincos_pos_embed_from_grid_odd_dimension(self) -> None:
        """Test get_1d_sincos_pos_embed_from_grid function with odd embed_dim."""
        embed_dim = 7

        with pytest.raises(AssertionError):
            get_1d_sincos_pos_embed_from_grid(embed_dim, np.array([0, 1]))

    def test_get_2d_sincos_pos_embed_from_grid(self) -> None:
        """Test get_2d_sincos_pos_embed_from_grid function."""
        embed_dim = 8
        grid = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        result = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        assert result.shape == (4, 8)
        assert result.dtype == np.float64

    def test_get_2d_sincos_pos_embed_from_grid_odd_dimension(self) -> None:
        """Test get_2d_sincos_pos_embed_from_grid function with odd embed_dim."""
        embed_dim = 7
        grid = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        with pytest.raises(AssertionError):
            get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    def test_get_1d_sincos_pos_embed(self) -> None:
        """Test get_1d_sincos_pos_embed function."""
        embed_dim = 8
        t_size = 4

        # Test without cls_token
        result = get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False)
        assert result.shape == (4, 8)

        # Test with cls_token
        result_with_cls = get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=True)
        assert result_with_cls.shape == (5, 8)
        # First row should be zeros for cls_token
        assert np.allclose(result_with_cls[0], 0)

    def test_get_2d_sincos_pos_embed(self) -> None:
        """Test get_2d_sincos_pos_embed function."""
        embed_dim = 8
        grid_size = 4

        # Test without cls_token
        result = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        assert result.shape == (16, 8)  # 4*4 = 16 positions

        # Test with cls_token
        result_with_cls = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
        assert result_with_cls.shape == (17, 8)  # 16 + 1 for cls_token
        # First row should be zeros for cls_token
        assert np.allclose(result_with_cls[0], 0)

    def test_get_3d_sincos_pos_embed(self) -> None:
        """Test get_3d_sincos_pos_embed function."""
        embed_dim = 32
        grid_size = 4
        t_size = 3

        # Test without cls_token
        result = get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False)
        assert result.shape == (48, embed_dim)  # 3*4*4 = 48 positions

        # Test with cls_token
        result_with_cls = get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=True)
        assert result_with_cls.shape == (49, embed_dim)  # 48 + 1 for cls_token
        # First row should be zeros for cls_token
        assert np.allclose(result_with_cls[0], 0)

    def test_get_3d_sincos_pos_embed_odd_dimension(self) -> None:
        """Test get_3d_sincos_pos_embed function with odd embed_dim."""
        embed_dim = 7
        grid_size = 4
        t_size = 3

        with pytest.raises(AssertionError):
            get_3d_sincos_pos_embed(embed_dim, grid_size, t_size)


class TestInterpolatePositionEmbedding:
    """Test cases for position embedding interpolation functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock checkpoint model
        self.checkpoint_model = {
            "vision_encoder.pos_embed": torch.randn(1, 17, 512),  # 1 + 4*4 positions
            "other_layer": torch.randn(1, 10, 512)
        }

        # Create mock model
        self.model = Mock()
        self.model.patch_embed.num_patches = 16  # 4*4 patches
        self.model.pos_embed.shape = (1, 17, 512)  # 1 + 4*4 positions
        self.model.T = 4  # 4 frames

    def test_interpolate_pos_embed_temporal_interpolation(self) -> None:
        """Test interpolate_pos_embed function with temporal interpolation."""
        # Set up model to expect different temporal size
        self.model.T = 8  # 8 frames instead of 4

        interpolate_pos_embed(
            self.checkpoint_model,
            self.model,
            orig_t_size=4,
            pos_name="vision_encoder.pos_embed"
        )

        # Check that temporal interpolation was applied
        pos_embed = self.checkpoint_model["vision_encoder.pos_embed"]
        assert pos_embed.shape[0] == 1  # batch size
        assert pos_embed.shape[2] == 512  # embedding dimension
        # The middle dimension should be different from the original due to temporal interpolation
        # It might be smaller or larger depending on the calculation
        assert pos_embed.shape[1] != 17

    def test_interpolate_pos_embed_spatial_interpolation(self) -> None:
        """Test interpolate_pos_embed function with spatial interpolation."""
        # Set up model to expect different spatial size
        self.model.patch_embed.num_patches = 64  # 8*8 patches instead of 4*4
        self.model.pos_embed.shape = (1, 65, 512)  # 1 + 8*8 positions
        self.model.T = 4

        interpolate_pos_embed(
            self.checkpoint_model,
            self.model,
            orig_t_size=4,
            pos_name="vision_encoder.pos_embed"
        )

        # Check that spatial interpolation was applied
        pos_embed = self.checkpoint_model["vision_encoder.pos_embed"]
        assert pos_embed.shape == (1, 65, 512)  # 1 + 8*8 positions

    def test_interpolate_pos_embed_no_interpolation_needed(self) -> None:
        """Test interpolate_pos_embed function when no interpolation is needed."""
        original_pos_embed = self.checkpoint_model["vision_encoder.pos_embed"].clone()

        interpolate_pos_embed(
            self.checkpoint_model,
            self.model,
            orig_t_size=4,
            pos_name="vision_encoder.pos_embed"
        )

        # Check that no changes were made
        pos_embed = self.checkpoint_model["vision_encoder.pos_embed"]
        assert torch.allclose(pos_embed, original_pos_embed)

    def test_interpolate_pos_embed_missing_key(self) -> None:
        """Test interpolate_pos_embed function with missing position embedding key."""
        # Remove the position embedding key
        del self.checkpoint_model["vision_encoder.pos_embed"]

        # Should not raise an error, just skip
        interpolate_pos_embed(
            self.checkpoint_model,
            self.model,
            orig_t_size=4,
            pos_name="vision_encoder.pos_embed"
        )


class TestInterpolatePositionEmbeddingInternVideo2:
    """Test cases for InternVideo2-specific position embedding interpolation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock checkpoint model
        self.checkpoint_model = {
            "pos_embed": torch.randn(1, 17, 512),  # 1 + 4*4 positions
            "clip_pos_embed": torch.randn(1, 17, 512),  # 1 + 4*4 positions
            "other_layer": torch.randn(1, 10, 512)
        }

        # Create mock model
        self.model = Mock()
        self.model.patch_embed.num_patches = 16  # 4*4 patches
        self.model.pos_embed.shape = (1, 17, 512)  # 1 + 4*4 positions
        self.model.num_frames = 8
        self.model.tubelet_size = 2

    def test_interpolate_pos_embed_internvideo2_temporal_interpolation(self) -> None:
        """Test interpolate_pos_embed_internvideo2 function with temporal interpolation."""
        # Set up model to expect different temporal size
        self.model.num_frames = 16  # 16 frames instead of 8
        self.model.tubelet_size = 2  # 16/2 = 8 temporal positions

        interpolate_pos_embed_internvideo2(
            self.checkpoint_model,
            self.model,
            orig_t_size=4
        )

        # Check that temporal interpolation was applied to both position embeddings
        pos_embed = self.checkpoint_model["pos_embed"]
        clip_pos_embed = self.checkpoint_model["clip_pos_embed"]
        assert pos_embed.shape[0] == 1  # batch size
        assert pos_embed.shape[2] == 512  # embedding dimension
        assert clip_pos_embed.shape[0] == 1  # batch size
        assert clip_pos_embed.shape[2] == 512  # embedding dimension
        # The middle dimension should be different from the original due to temporal interpolation
        assert pos_embed.shape[1] != 17
        assert clip_pos_embed.shape[1] != 17

    def test_interpolate_pos_embed_internvideo2_spatial_interpolation(self) -> None:
        """Test interpolate_pos_embed_internvideo2 function with spatial interpolation."""
        # Set up model to expect different spatial size
        self.model.patch_embed.num_patches = 64  # 8*8 patches instead of 4*4
        self.model.pos_embed.shape = (1, 65, 512)  # 1 + 8*8 positions
        self.model.num_frames = 8
        self.model.tubelet_size = 2

        interpolate_pos_embed_internvideo2(
            self.checkpoint_model,
            self.model,
            orig_t_size=4
        )

        # Check that spatial interpolation was applied
        pos_embed = self.checkpoint_model["pos_embed"]
        clip_pos_embed = self.checkpoint_model["clip_pos_embed"]
        assert pos_embed.shape == (1, 65, 512)  # 1 + 8*8 positions
        assert clip_pos_embed.shape == (1, 65, 512)  # 1 + 8*8 positions


class TestInterpolatePositionEmbeddingInternVideo2New:
    """Test cases for new InternVideo2 position embedding interpolation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock checkpoint model
        self.checkpoint_model = {
            "pos_embed": torch.randn(1, 17, 512),  # 1 + 4*4 positions
            "clip_pos_embed": torch.randn(1, 17, 512),  # 1 + 4*4 positions
            "other_layer": torch.randn(1, 10, 512)
        }

        # Create mock model
        self.model = Mock()
        self.model.patch_embed.num_patches = 16  # 4*4 patches
        self.model.pos_embed.shape = (1, 17, 512)  # 1 + 4*4 positions
        self.model.num_frames = 8
        self.model.tubelet_size = 2

    def test_interpolate_pos_embed_internvideo2_new_temporal_interpolation(self) -> None:
        """Test interpolate_pos_embed_internvideo2_new function with temporal interpolation."""
        # Set up model to expect different temporal size
        self.model.num_frames = 16  # 16 frames instead of 8
        self.model.tubelet_size = 2  # 16/2 = 8 temporal positions

        interpolate_pos_embed_internvideo2_new(
            self.checkpoint_model,
            self.model,
            orig_t_size=4
        )

        # Check that temporal interpolation was applied to both position embeddings
        pos_embed = self.checkpoint_model["pos_embed"]
        clip_pos_embed = self.checkpoint_model["clip_pos_embed"]
        assert pos_embed.shape[0] == 1  # batch size
        assert pos_embed.shape[2] == 512  # embedding dimension
        assert clip_pos_embed.shape[0] == 1  # batch size
        assert clip_pos_embed.shape[2] == 512  # embedding dimension
        # The middle dimension should be different from the original due to temporal interpolation
        assert pos_embed.shape[1] != 17
        assert clip_pos_embed.shape[1] != 17

    def test_interpolate_pos_embed_internvideo2_new_spatial_interpolation(self) -> None:
        """Test interpolate_pos_embed_internvideo2_new function with spatial interpolation."""
        # Set up model to expect different spatial size
        self.model.patch_embed.num_patches = 64  # 8*8 patches instead of 4*4
        self.model.pos_embed.shape = (1, 65, 512)  # 1 + 8*8 positions
        self.model.num_frames = 8
        self.model.tubelet_size = 2

        interpolate_pos_embed_internvideo2_new(
            self.checkpoint_model,
            self.model,
            orig_t_size=4
        )

        # Check that spatial interpolation was applied
        pos_embed = self.checkpoint_model["pos_embed"]
        clip_pos_embed = self.checkpoint_model["clip_pos_embed"]
        assert pos_embed.shape == (1, 65, 512)  # 1 + 8*8 positions
        assert clip_pos_embed.shape == (1, 65, 512)  # 1 + 8*8 positions
