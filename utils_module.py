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


def generate_link(model_name):
    return f'https://github.com/maurosonzogni/model_aaxl2/blob/main/xmi/{model_name}'



def extract_json_from_text(text):
    """
    Extracts a JSON object from a given text using regular expressions.
    The function searches for a string that looks like a JSON object (starts with '{' and ends with '}')
    and tries to convert it into a JSON object.

    Args:
    text (str): The text from which to extract the JSON object.

    Returns:
    dict: The extracted JSON object, or None if no valid JSON object is found.
    """
    # Using regular expressions to find a string that looks like a JSON object
    # This pattern looks for a string that starts with '{' and ends with '}'
    pattern = r'\{.*?\}'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            # Converts the extracted string into a JSON object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            # Handles the exception in case the string is not a valid JSON
            print("Unable to decode the JSON.")
            return None
    else:
        print("No JSON found in the text.")
        return None