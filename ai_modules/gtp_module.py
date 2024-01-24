# Note: The openai-python  support for Azure OpenAI is in preview.
# Note: This code sample requires OpenAI Python  version 0.28.1 or lower.
import os
import time

import openai
import tiktoken
from dotenv import load_dotenv

from utils_module import extract_json_from_text, load_config, remove_duplicate_words, remove_numbers, remove_substrings


def infer_topic_with_GPT(text, question, answer_format, role):

    # Load environment variables
    load_dotenv()

    openai.api_type = os.environ['API_TYPE']
    openai.api_base = os.environ['API_BASE']
    openai.api_version = os.environ['API_VERSION']
    openai.api_key = os.environ['API_KEY']
    
    # Configuration file path
    config_file_path = 'configurations/architectural_cluster_config.json'

    # Load configuration
    config = load_config(config_file_path)

    # Use default question and format if not provided
    if not question:
        question = os.environ['QUESTION']
    if not answer_format:
        answer_format = os.environ['ANSWER_FORMAT']

    if not isinstance(text, str):
        text = ' '.join(text)
    text = text.replace("_", " ").replace("->", " ").replace("-", " ").replace(".", " ")
    text = remove_numbers(text)
    text = remove_substrings(text, config['common_words_to_exclude'])
    # remove duplicate words to avoid useless token usage
    text = remove_duplicate_words(text)
    
    # Construct the complete question for the API
    complete_question = question + text + answer_format

    message_text = [{"role": role, "content": complete_question}]

    token = num_tokens_from_messages(message_text)
    max_response_tokens=800
    total_token_required = token + max_response_tokens
    
    if (total_token_required >= 16385):
        print("token size is too large"+ token)
    else:
        if (total_token_required > 4096):
            engine = os.environ['ENGINE_16K']
        else:
            engine = os.environ['ENGINE_4K']
        
        answer_content = openai.ChatCompletion.create(
        engine = engine,
        messages=message_text,
        temperature=0.7,
        max_tokens=max_response_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    

    try:
        if 'choices' in answer_content:
            answer_content = answer_content['choices']
            for choice in answer_content:
                answer_content_json = extract_json_from_text(
                    choice['message']['content'])
                if answer_content_json:
                    break
            print(answer_content_json)
        else:
            time.sleep(50)
            answer_content_json = extract_json_from_text(
                answer_content['content'])
            print(answer_content_json)
    except KeyError:
        answer_content_json = None
        print("Void")

    if not answer_content_json:
        return []

    # Return the topics list if it exists, otherwise return an empty list
    return answer_content_json.get('topics', [])



def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
