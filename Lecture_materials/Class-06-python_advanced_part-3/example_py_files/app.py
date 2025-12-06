# from operations.arithmaticOperations import add, subtract, multiply, divide
# from operations.stringOperations import countOccurences

import logging
from logger_config import set_logger

set_logger()
logger = logging.getLogger(__name__)

from operations import arithmaticOperations as aops
from operations import stringOperations as sops

logger.info("addition operation begins")
print(f"add 2 and 14: {aops.add(2, 14)}")

logger.info("subtraction operation begins")
print(f"subtract 5 from 26: {aops.subtract(26,5)}")

logger.info("division operation begins")
print(f"divide 25 by 5: {aops.divide(25,5)}")

string = "there is an apple tree in my garden"

string_without_space = string.replace(' ', '')

logger.info("count operation begins")
counts = sops.countOccurences(string_without_space, sort_output=True, ascending=False)

print(f"count = {counts}")