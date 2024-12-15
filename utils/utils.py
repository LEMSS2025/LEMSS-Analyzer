import hashlib

from constants.constants import HASH_TRUNCATION_LIMIT, HEX_BASE


def hash_id(id_str: str) -> int:
    """
    Hash a string ID to a stable integer ID with a maximum of 7 digits.

    :param id_str: The string ID to hash.

    :return: A stable integer hash of the string ID (7 digits max).
    """
    return int(hashlib.md5(id_str.encode()).hexdigest(), HEX_BASE) % HASH_TRUNCATION_LIMIT
