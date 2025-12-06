from logger import logging

def divide(a,b):
    logging.debug(f"Dividing {a} by {b}")
    try:
        result = a / b
        logging.info(f"Division successful")
        return result
    except ZeroDivisionError as e:
        logging.error("Attempted to divide by zero.")
        return None


logging.info("The code execution has started")
r = divide(10,0)
if r is not None:
    logging.info(f"Result = {r}")