import os
from bardapi import BardCookies
from dotenv import load_dotenv
from utils_module import extract_json_from_text

import time


def infer_topic_with_bard(text, question, answer_format):
    """
    Infers topics using Bard API with custom questions and formats.

    This function takes a piece of text and, optionally, a question and an answer format. It then queries the Bard API
    with the constructed question and extracts the relevant JSON content from the response, specifically returning the list of topics.

    Args:
    text (str): The text to analyze.
    question (str): A custom question to use for the analysis. If None, uses a default question from environment variables.
    answer_format (str): A custom format for the question. If None, uses a default format from environment variables.

    Returns:
    list: A list of topics inferred from the Bard API response.
    """
    # Load environment variables
    load_dotenv()

    # Setup cookies for Bard API authentication
    cookies = {
        "__Secure-1PSID": os.environ['Secure_1PSID'],
        "__Secure-1PSIDTS": os.environ['Secure_1PSIDTS'],
    }

    # Use default question and format if not provided
    if not question:
        question = os.environ['QUESTION']
    if not answer_format:
        answer_format = os.environ['ANSWER_FORMAT']

    # Initialize Bard API with cookies
    bard = BardCookies(cookie_dict=cookies)
    

    if not isinstance(text, str):
        text = ' '.join(text)

    # Construct the complete question for the API
    complete_question = question + text + answer_format
    time.sleep(4)
    # Get the answer from Bard API
    answer_content = bard.get_answer(complete_question)
    
    time.sleep(4)
    print(answer_content)

    try:
        if 'choices' in answer_content:
            answer_content = answer_content['choices']
            for choice in answer_content:
                answer_content_json = extract_json_from_text(choice['content'][0])
                if answer_content_json:
                    break
            print(answer_content_json)
        else:
            answer_content_json = extract_json_from_text(answer_content['content'])
            print(answer_content_json)
    except KeyError:
        answer_content_json = None
        print("Void")


    if not answer_content_json:
        return []
    
    # Return the topics list if it exists, otherwise return an empty list
    return answer_content_json.get('topics', [])

# Example usage
# topics = infer_topic_with_bard("your_text_here", "your_question_here", "your_answer_format_here")
# print(topics)
