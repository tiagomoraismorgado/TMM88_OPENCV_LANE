"""
Example module demonstrating basic unit testing patterns.
"""

def add_one(x: int) -> int:
    """
    Add 1 to the input number.
    
    Args:
        x: An integer to increment
        
    Returns:
        The input integer plus one
    """
    return x + 1


def test_add_one_returns_correct_result():
    """Test that add_one returns the expected value."""
    # Arrange
    input_value = 3
    expected = 4
    
    # Act
    result = add_one(input_value)
    
    # Assert
    assert result == expected, f"Expected {expected}, but got {result}"


def test_add_one_with_negative_number():
    """Test add_one with a negative input."""
    assert add_one(-5) == -4


def test_add_one_with_zero():
    """Test add_one with zero input."""
    assert add_one(0) == 1