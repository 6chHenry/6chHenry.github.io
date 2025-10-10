#  Write A Professional Docstring

## Google Style Docstring

```python
def multiply(a, b):
    """
    Multiply two numbers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: Product of a and b.
    """
    return a * b
print(multiply(3, 5))
```

## Numpy-Style Docstring

```python
def divide(a, b):
    """
    Divide two numbers.

    Parameters
    ----------
    a : float
        Dividend.
    b : float
        Divisor.

    Returns
    -------
    float
        Quotient of division.
    """
    if b == 0:
        raise ValueError("Division by zero not allowed.")
    return a / b
print(divide(6, 2))
```