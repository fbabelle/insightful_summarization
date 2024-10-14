import os
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
import logging

OPENAI_KEY = ""
CLAUDE_KEY = ""

# # Alternatively, load API keys from environment variables
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")

# Set the root directory
root_dir = '/home/'

# Set the list of domains
domains = ['news','article','scientific_research_paper']

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for other modules
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Constants for model selection
MAX_TOKENS = 4096
TOKEN_TRACKER: Dict[str, int] = {}

# Model selection function
def model_selection(
    model_name: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    output_json: bool = False,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.1
) -> str:
    kwargs = {}
    if output_json:
        kwargs['response_format'] = {"type": "json_object"}

    try:
        if model_name.startswith('gpt'):
            return _use_openai(model_name, messages or [{'role': 'system', 'content': prompt}], max_tokens, temperature, kwargs)
        elif model_name.startswith('claude'):
            return _use_anthropic(model_name, messages or [{'role': 'user', 'content': prompt}], max_tokens, temperature)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"Error in model_selection: {str(e)}")
        raise

def _use_openai(model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, kwargs: Dict) -> str:
    client = OpenAI(api_key=OPENAI_KEY, max_retries=10)
    res = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        **kwargs
    )
    return res.choices[0].message.content

def _use_anthropic(model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    client = Anthropic(api_key=CLAUDE_KEY, max_retries=10)
    res = client.messages.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS
    )
    return res.content[0].text