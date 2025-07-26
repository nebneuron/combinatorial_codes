"""
Unit tests for the combinatorial_codes package.

This module contains tests for the core functionality of the combinatorial_codes package,
including the CombinatorialCode class and its methods.
"""

import pytest
import numpy as np
from combinatorial_codes import CombinatorialCode, example_code
from combinatorial_codes.utils import intersections_via_cliques
from combinatorial_codes.correct_examples import (
    RANDOM_EXAMPLE_32_INTERSECTIONS,
    RANDOM_EXAMPLE_32_OBSTRUCTIONS
)


class TestExampleCodes:
    """Test suite for example codes and their expected properties."""

    def test_milo_example_simplicial_violators(self):
        """Test that 'example by Milo' produces the expected simplicial violators."""
        # Create the example code
        C = example_code("example by Milo")
        
        # Expected simplicial violators
        expected_violators = np.array([
            4, 5, 8, 12, 16, 20, 32, 64, 65,
            144, 256, 528, 1040, 1536, 2048, 2052, 2056, 2320,
            18432
        ], dtype=np.uint64)
        
        # Compute actual violators
        actual_violators = C.simplicial_violators()
        
        # Verify they match
        np.testing.assert_array_equal(
            np.sort(actual_violators), 
            np.sort(expected_violators),
            err_msg="Simplicial violators do not match expected values"
        )

    def test_milo_example_obstructions(self):
        """Test that 'example by Milo' produces the expected obstruction results."""
        # Create the example code
        C = example_code("example by Milo")
        
        # Expected obstruction results
        expected_is_complete = False
        expected_num_obstructions = 17
        
        # Compute actual obstructions
        actual_is_complete, actual_num_obstructions = C.Obstructions()
        
        # Verify they match
        assert actual_is_complete == expected_is_complete, \
            f"Expected is_complete={expected_is_complete}, got {actual_is_complete}"
        assert actual_num_obstructions == expected_num_obstructions, \
            f"Expected {expected_num_obstructions} obstructions, got {actual_num_obstructions}"

    def test_milo_example_consistency(self):
        """Test basic consistency properties of the Milo example."""
        # Create the example code
        C = example_code("example by Milo")
        
        # Get results
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Basic consistency checks
        assert isinstance(violators, np.ndarray), "Violators should be a numpy array"
        assert isinstance(is_complete, bool), "is_complete should be boolean"
        assert isinstance(num_obstructions, int), "num_obstructions should be integer"
        
        # For the Milo example, we know it should not be maximal intersection complete
        assert not is_complete, "Milo example should not be maximal intersection complete"
        
        # Should have positive number of obstructions
        assert num_obstructions > 0, "Should have some obstructions"
        
        # Should have some violators
        assert len(violators) > 0, "Should have some simplicial violators"


class TestOtherExampleCodes:
    """Test suite for other example codes to ensure basic functionality."""

    def test_eyes_example_basic(self):
        """Test that 'eyes' example creates a valid code."""
        C = example_code("eyes")
        
        # Basic sanity checks
        assert hasattr(C, 'words')
        assert hasattr(C, 'maximal_words')
        assert len(C.words) > 0
        assert len(C.maximal_words) > 0
        
        # Should be able to compute violators and obstructions without error
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Results should be consistent
        assert isinstance(violators, np.ndarray)
        assert isinstance(is_complete, bool)
        assert isinstance(num_obstructions, int)
        assert num_obstructions >= 0

    def test_open_not_closed_example_basic(self):
        """Test that 'open not closed' example creates a valid code."""
        C = example_code("open not closed")
        
        # Basic sanity checks
        assert hasattr(C, 'words')
        assert hasattr(C, 'maximal_words')
        assert len(C.words) > 0
        assert len(C.maximal_words) > 0
        
        # Should be able to compute violators and obstructions without error
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Results should be consistent
        assert isinstance(violators, np.ndarray)
        assert isinstance(is_complete, bool)
        assert isinstance(num_obstructions, int)
        assert num_obstructions >= 0

    def test_closed_not_open_example_basic(self):
        """Test that 'closed not open' example creates a valid code."""
        C = example_code("closed not open")
        
        # Basic sanity checks
        assert hasattr(C, 'words')
        assert hasattr(C, 'maximal_words')
        assert len(C.words) > 0
        assert len(C.maximal_words) > 0
        
        # Should be able to compute violators and obstructions without error
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Results should be consistent
        assert isinstance(violators, np.ndarray)
        assert isinstance(is_complete, bool)
        assert isinstance(num_obstructions, int)
        assert num_obstructions >= 0


class TestCombinatorialCodeClass:
    """Test suite for the CombinatorialCode class functionality."""

    def test_combinatorial_code_creation(self):
        """Test that CombinatorialCode can be created with different inputs."""
        # Test with simple vectors
        vectors = [[], [1], [2], [1, 2]]
        C = CombinatorialCode(vectors)
        
        assert hasattr(C, 'words')
        assert hasattr(C, 'maximal_words')
        assert len(C.words) >= len(vectors)  # May include additional words
        
    def test_empty_set_detection(self):
        """Test that codes correctly detect presence of empty set."""
        # Code with empty set
        vectors_with_empty = [[], [1], [2]]
        C_with_empty = CombinatorialCode(vectors_with_empty)
        assert C_with_empty.has_empty_set()
        
        # Code without empty set
        vectors_without_empty = [[1], [2], [1, 2]]
        C_without_empty = CombinatorialCode(vectors_without_empty)
        assert not C_without_empty.has_empty_set()

    def test_code_representation(self):
        """Test that codes have proper string representation."""
        vectors = [[], [1], [2]]
        C = CombinatorialCode(vectors)
        
        # Should have a string representation
        repr_str = repr(C)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


class TestRandomCodes:
    """Test suite for randomly generated codes."""

    def test_bernoulli_random_code_creation(self):
        """Test that Bernoulli random codes can be created."""
        from combinatorial_codes.examples import bernoulli_random_code
        
        # Create a small random code
        C = bernoulli_random_code(n_bits=5, Nwords=10, p=0.5)
        
        # Basic sanity checks
        assert isinstance(C, CombinatorialCode)
        assert hasattr(C, 'words')
        assert hasattr(C, 'maximal_words')
        
        # Should be able to compute violators
        violators = C.simplicial_violators()
        assert isinstance(violators, np.ndarray)
        
        # Test obstructions only if the code has an empty set
        if C.has_empty_set():
            is_complete, num_obstructions = C.Obstructions()
            assert isinstance(is_complete, bool)
            assert isinstance(num_obstructions, int)
        else:
            # Should raise ValueError for codes without empty set
            with pytest.raises(ValueError, match="does not have an empty set"):
                C.Obstructions()


class TestPackageIntegration:
    """Test suite for package integration and performance features."""

    def test_c_extension_status(self):
        """Test that C extension status checking works."""
        import combinatorial_codes as cc
        
        # Should be able to check status without error
        status = cc.quick_status()
        assert isinstance(status, str)
        assert len(status) > 0
        
        # Detailed status check should also work
        # This function prints output, so we just verify it doesn't crash
        try:
            cc.check_c_extension_status()
        except Exception as e:
            pytest.fail(f"C extension status check failed: {e}")

    def test_performance_modes(self):
        """Test that both performance modes (C and Numba) work."""
        from combinatorial_codes import utils
        
        # Check if we have access to the tf module (C extensions)
        has_c_extensions = utils.tf is not None
        
        # Create a small test case
        C = example_code("eyes")
        
        # These should work regardless of which performance mode is active
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Results should be valid
        assert isinstance(violators, np.ndarray)
        assert isinstance(is_complete, bool)
        assert isinstance(num_obstructions, int)
        
        print(f"Tests running with C extensions: {has_c_extensions}")


class TestAddEmptyWordMethod:
    """Test suite for the add_empty_word method."""

    def test_add_empty_word_to_empty_code(self):
        """Test adding empty word to an empty code."""
        C = CombinatorialCode([])
        
        # Initially should not have empty set
        assert not C.has_empty_set()
        assert C.n_words == 0
        
        # Add empty word
        result = C.add_empty_word()
        
        # Should succeed and update correctly
        assert result == True
        assert C.has_empty_set()
        assert C.n_words == 1
        assert 0 in C.words
        assert C.min_size == 0
        assert C.max_size == 0

    def test_add_empty_word_to_existing_code(self):
        """Test adding empty word to a code with existing words."""
        C = CombinatorialCode([[1], [2], [1, 2]])
        
        # Store initial state
        initial_n_words = C.n_words
        initial_maximal_words = C.maximal_words.copy()
        
        # Should not have empty set initially
        assert not C.has_empty_set()
        assert C.min_size > 0
        
        # Add empty word
        result = C.add_empty_word()
        
        # Should succeed and update correctly
        assert result == True
        assert C.has_empty_set()
        assert C.n_words == initial_n_words + 1
        assert 0 in C.words
        assert C.min_size == 0
        
        # Maximal words should remain unchanged
        np.testing.assert_array_equal(C.maximal_words, initial_maximal_words)

    def test_add_empty_word_when_already_present(self):
        """Test that adding empty word when already present does nothing."""
        C = CombinatorialCode([[], [1], [2]])
        
        # Should already have empty set
        assert C.has_empty_set()
        initial_n_words = C.n_words
        
        # Try to add empty word again
        result = C.add_empty_word()
        
        # Should fail (return False) and not change anything
        assert result == False
        assert C.n_words == initial_n_words
        assert C.has_empty_set()

    def test_add_empty_word_updates_attributes_correctly(self):
        """Test that all attributes are updated correctly when adding empty word."""
        C = CombinatorialCode([[1], [2], [1, 2], [3]])
        
        # Add empty word
        C.add_empty_word()
        
        # Check unique_sizes includes 0
        assert 0 in C.unique_sizes
        
        # Check indices_by_size is correct
        for size in C.unique_sizes:
            indices = C.indices_by_size[size]
            words_of_this_size = C.words[indices]
            actual_sizes = [w.bit_count() for w in words_of_this_size]
            assert all(s == size for s in actual_sizes), f"Size mismatch for size {size}"
        
        # Check that empty word is at size 0
        size_0_indices = C.indices_by_size[0]
        assert len(size_0_indices) == 1
        assert C.words[size_0_indices[0]] == 0

    def test_add_empty_word_preserves_functionality(self):
        """Test that adding empty word preserves code functionality."""
        C = CombinatorialCode([[1], [2], [1, 2]])
        
        # Add empty word
        C.add_empty_word()
        
        # Should still be able to compute violators and obstructions
        violators = C.simplicial_violators()
        is_complete, num_obstructions = C.Obstructions()
        
        # Results should be valid
        assert isinstance(violators, np.ndarray)
        assert isinstance(is_complete, bool)
        assert isinstance(num_obstructions, int)
        assert num_obstructions >= 0


class TestRegressionResults:
    """Test suite for regression testing against known correct results."""

    def test_random_example_32_intersections(self):
        """Test that 'random example 32' produces the expected intersections."""
        # Load the example code
        C = example_code("random example 32")
        
        # Compute intersections
        actual_intersections = intersections_via_cliques(C.maximal_words)
        
        # Compare with expected results
        np.testing.assert_array_equal(
            actual_intersections, 
            RANDOM_EXAMPLE_32_INTERSECTIONS,
            err_msg="Random example 32 intersections do not match expected values"
        )
        
        # Additional consistency checks
        assert len(actual_intersections) == len(RANDOM_EXAMPLE_32_INTERSECTIONS)
        assert actual_intersections.dtype == RANDOM_EXAMPLE_32_INTERSECTIONS.dtype

    def test_random_example_32_obstructions(self):
        """Test that 'random example 32' produces the expected obstruction results."""
        # Load the example code
        C = example_code("random example 32")
        
        # Compute obstructions
        actual_is_complete, actual_num_obstructions = C.Obstructions()
        
        # Compare with expected results
        expected_is_complete, expected_num_obstructions = RANDOM_EXAMPLE_32_OBSTRUCTIONS
        
        assert actual_is_complete == expected_is_complete, \
            f"Expected is_complete={expected_is_complete}, got {actual_is_complete}"
        assert actual_num_obstructions == expected_num_obstructions, \
            f"Expected {expected_num_obstructions} obstructions, got {actual_num_obstructions}"

    def test_random_example_32_consistency(self):
        """Test consistency between intersections and obstructions for random example 32."""
        # Load the example code
        C = example_code("random example 32")
        
        # Compute both results
        intersections = intersections_via_cliques(C.maximal_words)
        is_complete, num_obstructions = C.Obstructions()
        
        # Basic consistency checks
        assert isinstance(intersections, np.ndarray), "Intersections should be a numpy array"
        assert isinstance(is_complete, bool), "is_complete should be boolean"
        assert isinstance(num_obstructions, int), "num_obstructions should be integer"
        
        # For this example, we know it should not be maximal intersection complete
        assert not is_complete, "Random example 32 should not be maximal intersection complete"
        
        # Should have positive number of obstructions and intersections
        assert num_obstructions > 0, "Should have some obstructions"
        assert len(intersections) > 0, "Should have some intersections"
        
        # Verify intersections are properly sorted and unique
        assert np.all(intersections[:-1] < intersections[1:]), "Intersections should be sorted"
        assert len(intersections) == len(np.unique(intersections)), "Intersections should be unique"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
