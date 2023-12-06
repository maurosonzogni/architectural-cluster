import json
import os
import re

def load_config(config_file_path):
    """
    Load configuration settings from a JSON file.

    Args:
        config_file_path (str): The file path to the configuration file.

    Returns:
        dict or None: A dictionary containing the configuration settings if the file is successfully
                      loaded and parsed. Returns None if the file is not found or if there is an error
                      decoding the JSON content.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        json.JSONDecodeError: If there is an error decoding the JSON content of the configuration file.
    """
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error reading configuartion file: {config_file_path}")
        return None
    

def create_parent_folders(file_path):
    """
    Create the folder structure for the parent directory of the specified file path.

    Args:
        file_path (str): The file path for which to create the parent folder structure.
    """
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)


def remove_numbers(sentence):
    """
    Removes numeric digits from the given sentence.

    Args:
        sentence (str): The input sentence containing alphanumeric characters.

    Returns:
        str: A new string with numeric digits removed.

    Example:
        >>> remove_numbers("Hello123 World456")
        'Hello World'
    """
    sentence_without_numbers = re.sub(r'\d+', '', sentence)
    return sentence_without_numbers


def remove_substrings(input_string, substrings_to_remove):
    """
    Removes all occurrences of substrings in the input array from the given string.
    Case-insensitive.

    Args:
        input_string (str): The input string to be modified.
        substrings_to_remove (list): An array of strings to be removed from the input string.

    Returns:
        str: A new string with specified substrings removed.

    Example:
        >>> remove_substrings("Hello World, Hello Universe", ["hello", "universe"])
        ' World, '
    """
    input_string_lower = input_string.lower()
    for substring in substrings_to_remove:
        input_string_lower = input_string_lower.replace(substring.lower(), '')
    return input_string_lower