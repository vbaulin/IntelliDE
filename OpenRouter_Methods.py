# OpenRouter_Methods.py (Handles BOTH OpenRouter and Gemini, with Retries)
import os
from openai import OpenAI
import logging
import json
from pathlib import Path
import time
from typing import Optional, Literal
import backoff
from utils import load_prompt  # Corrected import
from google import genai #Corrected import
import re

def setup_logging(name=__name__):
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

def get_openrouter_client():
    """Initialize OpenRouter client"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

# Exponential backoff decorator for handling rate limits
@backoff.on_exception(
    backoff.expo,
    (Exception),  
    max_tries=5,  # Maximum number of retries
    max_time=300,  # Maximum total time to try in seconds
    giveup=lambda e: not (isinstance(e, Exception) and
        ('rate' in str(e).lower() or 'timeout' in str(e).lower())),
)
def get_openrouter_response_with_retry(client, prompt: str,
                                     max_retries: int = 30, # Now using max_retries
                                     retry_delay: int = 5) -> Optional[str]: # Using retry_delay
    """
    Get response from OpenRouter API with retry logic

    Args:
        client: OpenRouter client
        prompt: Input prompt
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Response content or None if all retries failed
    """
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                #model="deepseek/deepseek-r1-distill-llama-70b:free",
                model="google/gemini-2.0-flash-thinking-exp:free",
                #model="google/gemini-2.0-pro-exp-02-05:free",
                messages=[{"role": "user", "content": prompt}]
            )

            # Check if response is valid
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content

            logging.warning(f"Empty or invalid response received, attempt {retries + 1}/{max_retries}")
            print(response)

        except Exception as e:
            logging.error(f"❌ Error getting OpenRouter response (attempt {retries + 1}/{max_retries}): {e}")

            # Check if it's a rate limit error
            if 'rate' in str(e).lower():
                logging.info(f"Rate limit hit. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"❌ Unexpected error: {e}")
                if '400' in str(e):
                    logging.error("Request details for debugging:")
                    logging.error(f"Prompt length: {len(prompt)}")
                    logging.error(f"First 100 chars of prompt: {prompt[:100]}...")

        retries += 1
        time.sleep(retry_delay)

    logging.error("❌ All retry attempts failed")
    return None

def get_openrouter_response(client: OpenAI, prompt: str) -> str:
    """
    Main response function that handles retries and validates output
    """
    response = get_openrouter_response_with_retry(client, prompt, max_retries=30, retry_delay=5)

    if response is None:
        raise Exception("❌ Failed to get valid response after all retries")

    return response

@backoff.on_exception(
     backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: not (isinstance(e, Exception) and
        ('rate' in str(e).lower() or 'timeout' in str(e).lower() or 'internal' in str(e).lower())),
)
def get_gemini_response_with_retry(prompt: str) -> str:
    """Gets response from Gemini API with retry logic, using backoff."""
    google_api_key = os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=google_api_key) #CORRECT
    try:
        response = client.models.generate_content(model="gemini-pro", contents=prompt) # Use gemini-pro
        return response.text
    except Exception as e:
        logging.error(f"❌ Error getting Google Gemini response: {e}")
        raise #Re-raise exception

def get_gemini_response(prompt:str) -> str:
    response = get_gemini_response_with_retry(prompt)
    if response is None:
        raise Exception("Failed to get a valid response after retries.")
    return response

# --- Unified LLM Response Function ---
def get_llm_response(prompt: str, api_choice: Literal["google", "openrouter"]) -> str:
    """Gets LLM response, handling API choice and cleaning."""
    setup_logging() # call setup logging
    if api_choice == "openrouter":
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        client = get_openrouter_client()
        response_text = get_openrouter_response(client, prompt)

    elif api_choice == "google":
        response_text = get_gemini_response(prompt) # call the new gemini function.
    else:
        raise ValueError(f"Invalid API choice: {api_choice}")

    if not response_text:
        return ""

    return response_text.strip() # Return the raw response

def save_markdown_report(content: str, filepath: Path, title: str):
    """Save report in markdown format"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)

    except Exception as e:
        logging.error(f"❌ Error saving markdown report: {e}")
        raise

def save_json_report(content: str, filepath: Path, metadata: dict):
    """Save report in JSON format"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": metadata,
            "content": content
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logging.error(f"❌ Error saving JSON report: {e}")
        raise
