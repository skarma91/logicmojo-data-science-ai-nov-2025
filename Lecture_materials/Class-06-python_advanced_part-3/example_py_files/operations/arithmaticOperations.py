import logging

logger = logging.getLogger(__name__)


def add(a: int|float, b: int|float )-> int|float:
    """
    Adding two numbers.

    Args:
        - a : (int or float) [required] first number
        - b : (int or float) [required] second number

    Return:
        - result : (int or float) summation of a and b
    """
    result = a + b
    logger.info(f"The result of add function : {result}")
    return result


def subtract(a: int|float, b: int|float )-> int|float:
    """
    Subtracting two numbers.

    Args:
        - a : (int or float) [required] first number
        - b : (int or float) [required] second number

    Return:
        - result : (int or float) subtraction of b from a
    """
    result = a - b
    logger.info(f"The result of subtract function : {result}")
    return result


def multiply(a: int|float, b: int|float )-> int|float:
    """
    Adding two numbers.

    Args:
        - a : (int or float) [required] first number
        - b : (int or float) [required] second number

    Return:
        - result : (int or float) multiplication of a and b
    """
    result = a * b
    logger.info(f"The result of multiply function : {result}")
    return result


def divide(a: int|float, b: int|float )-> float|None:
    """
    Adding two numbers.

    Args:
        - a : (int or float) [required] first number
        - b : (int or float) [required] second number

    Return:
        - result : (float | None) division of a by b
    """
    try:
        result = a / b
    except ZeroDivisionError as e:
        logger.error(f"Error is division function. The error message={e}")
        return None
    else:
        logger.info(f"result of division function: {result}")
        return result
    
if __name__ == "__main__":
    print(add(2, 3))
    print(multiply(5, 10))
    print(divide(9, 3))