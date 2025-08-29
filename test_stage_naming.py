#!/usr/bin/env python3
"""
Minimal test for stage naming validation.
"""
import re
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def _validate_stage_name(name: str) -> None:
    """Validate that a stage name follows snake_case convention."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Stage name must be snake_case, got '{name}'")

def test_stage_naming_validation():
    """Test the stage name validation function."""
    
    # Test valid snake_case names
    valid_names = [
        "valid_name",
        "snake_case_name", 
        "name_with_123",
        "simple",
        "a",
        "word_word_word",
        "deduplication_stage",
        "common_crawl_main_pipeline"
    ]
    
    for name in valid_names:
        try:
            _validate_stage_name(name)
            print(f"✓ '{name}' is valid")
        except ValueError as e:
            print(f"✗ '{name}' should be valid but failed: {e}")
            return False
    
    # Test invalid names
    invalid_names = [
        "CamelCase",
        "PascalCase", 
        "InvalidCamelCase",
        "RemovalStage",
        "PairwiseCosineSimilarityStage",
        "KMeansStage",
        "DuplicatesRemovalStage",
        "123invalid",
        "_starts_with_underscore",
        "has-dashes",
        "has spaces",
        "has.dots",
        "ALLCAPS",
        "",
        "Mixed_Case_Name"
    ]
    
    for name in invalid_names:
        try:
            _validate_stage_name(name)
            print(f"✗ '{name}' should be invalid but passed")
            return False
        except ValueError:
            print(f"✓ '{name}' correctly rejected")
    
    print("\n✅ All validation tests passed!")
    return True

if __name__ == "__main__":
    success = test_stage_naming_validation()
    sys.exit(0 if success else 1)