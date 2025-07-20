"""
Unit tests for the combinatorial_codes package.

This module contains tests for the core functionality of the combinatorial_codes package,
including the CombinatorialCode class and its methods.
"""

import pytest
import numpy as np
from combinatorial_codes import CombinatorialCode, example_code


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


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
