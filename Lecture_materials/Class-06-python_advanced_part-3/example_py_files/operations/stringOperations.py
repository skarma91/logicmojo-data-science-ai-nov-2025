from logger_config import logging

logger = logging.getLogger(__name__)


def countOccurences(s: str, sort_output: bool=True, ascending: bool=True)->dict['str', int]:
    """
    This function counts the number of occurrences of individual characters 
    In the string

    Args:
        - s : (str) [required] The input string
        - sort_output : (bool) [optional] Whether the count dictionary will be sorted or not. 
                        Default is True.
        - ascending : (bool) [optional] Whether the sorted dictionary will be in ascending
                      order or not. Default is True.
    
    Returns:
        - counts : (dict) A dictionary containing the character and 
                    its count as key-value pairs. It will be sorted
                    if `sorted` arg is True. The order of sorting
                    will be determined by the `ascending` arg.

    Example:
    s = "aabcdbb", sorted=False

    count = {
        'a': 2,
        'b': 3,
        'c': 1,
        'd': 1
    }
    """

    counts = {}

    for c in s:
        counts[c] = counts.get(c, 0) + 1

    if sort_output:
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse= not ascending))

    logger.debug("The method runs succesfully")
    logger.info(f"The counts are : {counts}")
    return counts