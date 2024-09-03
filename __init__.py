import os
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Load API keys from environment variables
# ZEROAI_KEY = os.getenv("ZEROAI_API_KEY")
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")

# Set environment variables for other modules
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

MAX_TOKENS = 4096
TOKEN_TRACKER: Dict[str, int] = {}

def model_selection(
    model_name: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    output_json: bool = False,
    temperature: float = 0.1
) -> str:
    kwargs = {}
    if output_json:
        kwargs['response_format'] = {"type": "json_object"}

    try:
        if model_name.startswith('gpt'):
            return _use_openai(model_name, messages or [{'role': 'system', 'content': prompt}], temperature, kwargs)
        elif model_name.startswith('claude'):
            return _use_anthropic(model_name, messages or [{'role': 'user', 'content': prompt}], temperature)
        elif model_name.startswith('zero'):
            return _use_zeroai(messages or [{'role': 'system', 'content': prompt}], temperature, kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"Error in model_selection: {str(e)}")
        raise

def _use_openai(model_name: str, messages: List[Dict[str, str]], temperature: float, kwargs: Dict) -> str:
    client = OpenAI(api_key=OPENAI_KEY, max_retries=10)
    res = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        **kwargs
    )
    return res.choices[0].message.content

def _use_anthropic(model_name: str, messages: List[Dict[str, str]], temperature: float) -> str:
    client = Anthropic(api_key=CLAUDE_KEY, max_retries=10)
    res = client.messages.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS
    )
    return res.content[0].text

def _use_zeroai(messages: List[Dict[str, str]], temperature: float, kwargs: Dict) -> str:
    client = OpenAI(api_key=ZEROAI_KEY, base_url="https://api.zeroai.link/v1", max_retries=10)
    res = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        **kwargs
    )
    return res.choices[0].message.content