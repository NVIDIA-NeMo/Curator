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

import json
from pathlib import Path

import pytest
import ray

from nemo_curator.stages.deduplication.id_generator import (
    IdGeneratorBase,
    create_id_generator_actor,
    get_id_generator_actor,
    kill_id_generator_actor,
    write_id_generator_to_disk,
)


@pytest.fixture
def sample_batch_registry():
    """Sample batch registry for testing."""
    return {
        "batch1-hash": (0, 99),
        "batch2-hash": (100, 199),
        "batch3-hash": (200, 299),
    }


class TestIdGeneratorBase:
    """Test the IdGeneratorBase class directly (not as Ray actor)."""

    def test_init_default(self):
        """Test IdGeneratorBase initialization with defaults."""
        generator = IdGeneratorBase()
        assert generator.next_id == 0
        assert generator.batch_registry == {}

        """Test IdGeneratorBase initialization with parameters."""
        generator = IdGeneratorBase(start_id=100, batch_registry=sample_batch_registry)
        assert generator.next_id == 100
        assert generator.batch_registry == sample_batch_registry

    def test_hash_files_single(self):
        """Test hashing a single file."""
        generator = IdGeneratorBase()
        hash1 = generator.hash_files("file1.txt")
        hash2 = generator.hash_files("file1.txt")
        hash3 = generator.hash_files("file2.txt")

        # Same file should produce same hash
        assert hash1 == hash2
        # Different files should produce different hashes
        assert hash1 != hash3
        # Hash should be a string
        assert isinstance(hash1, str)

        """Test hashing a list of files."""
        generator = IdGeneratorBase()
        files = ["file1.txt", "file2.txt", "file3.txt"]
        hash_list1 = generator.hash_files(files)
        hash_list2 = generator.hash_files(files)
        hash_list3 = generator.hash_files(["file1.txt", "file2.txt"])

        # Same file list should produce same hash
        assert hash_list1 == hash_list2
        # Different file lists should produce different hashes
        assert hash_list1 != hash_list3

    def test_register_batch_new(self):
        """Test registering a new batch."""
        generator = IdGeneratorBase()
        files = ["file1.txt", "file2.txt"]
        count = 50

        start_id = generator.register_batch(files, count)

        assert start_id == 0
        assert generator.next_id == 50

        # Check batch registry
        batch_hash = generator.hash_files(files)
        assert batch_hash in generator.batch_registry
        assert generator.batch_registry[batch_hash] == (0, 49)

        # Check get_batch_range
        start_id, end_id = generator.get_batch_range(files=files, key=None)
        assert start_id == 0
        assert end_id == 49

    def test_register_batch_existing(self):
        """Test registering an already registered batch."""
        generator = IdGeneratorBase()
        files = ["file1.txt", "file2.txt"]
        count = 50

        # Register first time
        start_id1 = generator.register_batch(files, count)

        # Register same batch again
        start_id2 = generator.register_batch(files, count)

        # Should return same start ID
        assert start_id1 == start_id2
        # next_id should not advance further
        assert generator.next_id == 50

    def test_register_multiple_batches(self):
        """Test registering multiple different batches."""
        generator = IdGeneratorBase()

        # Register first batch
        start_id1 = generator.register_batch(["file1.txt"], 30)
        assert start_id1 == 0
        assert generator.next_id == 30

        # Register second batch
        start_id2 = generator.register_batch(["file2.txt"], 20)
        assert start_id2 == 30
        assert generator.next_id == 50

        # Register third batch
        start_id3 = generator.register_batch(["file3.txt"], 10)
        assert start_id3 == 50
        assert generator.next_id == 60

        # Check get_batch_range
        assert generator.get_batch_range(files=["file1.txt"], key=None) == (0, 29)
        assert generator.get_batch_range(files=["file2.txt"], key=None) == (30, 49)
        assert generator.get_batch_range(files=["file3.txt"], key=None) == (50, 59)

    def test_get_batch_range_by_key(self):
        """Test getting batch range by key."""
        generator = IdGeneratorBase()
        files = ["file1.txt", "file2.txt"]

        # Register batch
        generator.register_batch(files, 50)

        # Get range by key
        batch_hash = generator.hash_files(files)
        start_id, end_id = generator.get_batch_range(files=None, key=batch_hash)
        assert start_id == 0
        assert end_id == 49

    def test_get_batch_range_nonexistent(self):
        """Test getting range for non-existent batch."""
        generator = IdGeneratorBase()

        with pytest.raises(KeyError):
            generator.get_batch_range(files=["nonexistent.txt"], key=None)

    def test_to_disk(self, tmp_path: Path):
        """Test saving IdGeneratorBase state to disk."""
        generator = IdGeneratorBase(start_id=100)
        generator.register_batch(["file1.txt"], 50)
        generator.register_batch(["file2.txt"], 30)

        temp_file = tmp_path / "test_state.json"
        generator.to_disk(str(temp_file))

        # Verify file was created and has correct content
        assert temp_file.exists()

        with open(temp_file) as f:
            data = json.load(f)

        assert data["next_id"] == 180
        assert len(data["batch_registry"]) == 2

    def test_from_disk(self, tmp_path: Path):
        """Test loading IdGeneratorBase state from disk."""
        # Create initial generator and save to disk
        original_generator = IdGeneratorBase(start_id=100)
        original_generator.register_batch(["file1.txt"], 50)
        original_generator.register_batch(["file2.txt"], 30)

        temp_file = tmp_path / "test_state.json"
        original_generator.to_disk(str(temp_file))

        # Load from disk
        loaded_generator = IdGeneratorBase.from_disk(str(temp_file))

        # Verify state was restored correctly
        assert loaded_generator.next_id == 180
        assert len(loaded_generator.batch_registry) == 2

        # Verify batch registry works
        start_id, end_id = loaded_generator.get_batch_range(files=["file1.txt"], key=None)
        assert start_id == 100
        assert end_id == 149


class TestIdGeneratorActor:
    """Test the IdGenerator as a Ray actor."""

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_actor_lifecycle(self, tmp_path: Path, reuse_actor: bool = False):
        # We start a new ray context and interact with the actor
        with ray.init(ignore_reinit_error=True):
            actor = get_id_generator_actor()

            result_file1 = ray.get(actor.register_batch.remote(["file1.txt"], 10))
            assert result_file1 == 0
            result_file2 = ray.get(actor.register_batch.remote(["file2.txt"], 5))
            assert result_file2 == 10

            assert ray.get(actor.get_batch_range.remote(files=["file1.txt"], key=None)) == (0, 9)
            assert ray.get(actor.get_batch_range.remote(files=["file2.txt"], key=None)) == (10, 14)

            hash_key = ray.get(actor.hash_files.remote(["file1.txt"]))
            assert ray.get(actor.get_batch_range.remote(files=None, key=hash_key)) == (0, 9)

            hash_key = ray.get(actor.hash_files.remote(["file2.txt"]))
            assert ray.get(actor.get_batch_range.remote(files=None, key=hash_key)) == (10, 14)

            # Raises KeyError if the file is not registered
            with pytest.raises(KeyError):
                ray.get(actor.get_batch_range.remote(files=["file3.txt"], key=None))

            # We save the actor state to disk
            temp_file = tmp_path / "test_state.json"

        # Outside the ray context
        # we write the actor state to disk
        if reuse_actor:
            write_id_generator_to_disk(str(temp_file))
        # we kill the actor
        kill_id_generator_actor()

        # We create a new actor
        if reuse_actor:
            create_id_generator_actor(str(temp_file))
        else:
            create_id_generator_actor()

        # We start a new ray context and interact with the actor
        with ray.init(ignore_reinit_error=True):
            actor = get_id_generator_actor()
            result_file2 = ray.get(actor.register_batch.remote(["file2.txt"], 5))
            result_file1 = ray.get(actor.register_batch.remote(["file1.txt"], 10))
            if reuse_actor:
                assert result_file1 == 0
                assert result_file2 == 10
            else:
                assert result_file2 == 0
                assert result_file1 == 5
