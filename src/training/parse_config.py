"""
parse_config.py

Utility functions for parsing configuration files.

Functions:
- parse_config: Parse the configuration file.
"""

import configparser
from typing import Dict


def parse_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the configuration file.

    Args:
    - config_path (str): Path to the configuration file.

    Returns:
    - config (Dict[str, Dict[str, str]]): Configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    dict_config = {}
    for section in config.sections():
        dict_config[section] = dict(config.items(section))

    return dict_config
