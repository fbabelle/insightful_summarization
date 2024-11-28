import os
from typing import Dict, List, Optional, Any
from openai import OpenAI
from anthropic import Anthropic
import tiktoken
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

OPENAI_KEY = ""
CLAUDE_KEY = ""

# # Alternatively, load API keys from environment variables
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")

# Set the root directory
root_dir = '/home/insightful_summarization/'

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


# Dataclass for cache configuration
@dataclass
class CacheConfig:
    max_size_mb: int
    max_tokens: int
    ttl: int  # Time-to-live in seconds

class TokenAwareCache:
    def __init__(self, config: CacheConfig):
        self.cache: Dict[str, Any] = {}
        self.token_counts: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self.config = config
        self.current_tokens = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if time.time() - self.access_times[key] > self.config.ttl:
                self._evict(key)
                return None
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, value: Any, token_count: int):
        if token_count > self.config.max_tokens:
            return

        while self.current_tokens + token_count > self.config.max_tokens:
            self._evict_lru()

        if key in self.cache:
            self._evict(key)

        self.cache[key] = value
        self.token_counts[key] = token_count
        self.access_times[key] = time.time()
        self.current_tokens += token_count

    def _evict(self, key: str):
        if key in self.cache:
            self.current_tokens -= self.token_counts[key]
            del self.cache[key]
            del self.token_counts[key]
            del self.access_times[key]

    def _evict_lru(self):
        if not self.access_times:
            return
        lru_key = min(self.access_times, key=self.access_times.get)
        self._evict(lru_key)

# Then create cache instances for different components:
CACHE_CONFIGS = {
    'classification': CacheConfig(max_size_mb=256, max_tokens=100_000, ttl=3600),  # 1 hour TTL
    'summary': CacheConfig(max_size_mb=512, max_tokens=200_000, ttl=7200),        # 2 hours TTL
    'fact_check': CacheConfig(max_size_mb=1024, max_tokens=300_000, ttl=3600)     # 1 hour TTL
}

classification_cache = TokenAwareCache(CACHE_CONFIGS['classification'])
summary_cache = TokenAwareCache(CACHE_CONFIGS['summary'])
fact_check_cache = TokenAwareCache(CACHE_CONFIGS['fact_check'])

# Helper functions for cache management
def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Returns the encoding for the specified name."""
    return tiktoken.get_encoding(encoding_name)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(string))



# Available document type domains
domains = ["scientific_research_paper", "news", "article"]

class AnalysisDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

@dataclass
class DepthConfig:
    chunk_size: int
    overlap: float
    voting_threshold: float
    verification_depth: int
    temporal_detail: bool
    max_summary_tokens: int
    parallel_processing: bool

class AdaptiveDepthManager:
    def __init__(self):
        self.depth_configs = {
            AnalysisDepth.QUICK: DepthConfig(
                chunk_size=2000,
                overlap=0.05,
                voting_threshold=0.6,
                verification_depth=1,
                temporal_detail=False,
                max_summary_tokens=500,
                parallel_processing=False
            ),
            AnalysisDepth.STANDARD: DepthConfig(
                chunk_size=1000,
                overlap=0.1,
                voting_threshold=0.7,
                verification_depth=2,
                temporal_detail=True,
                max_summary_tokens=1000,
                parallel_processing=True
            ),
            AnalysisDepth.DEEP: DepthConfig(
                chunk_size=500,
                overlap=0.2,
                voting_threshold=0.8,
                verification_depth=3,
                temporal_detail=True,
                max_summary_tokens=2000,
                parallel_processing=True
            )
        }

    def determine_depth(self, 
                       document_type: str,
                       content_length: int,
                       time_constraint: Optional[int] = None) -> AnalysisDepth:
        """Determine appropriate analysis depth based on document characteristics."""
        if time_constraint:
            if time_constraint < 10:
                return AnalysisDepth.QUICK
            elif time_constraint < 30:
                return AnalysisDepth.STANDARD

        if document_type == "news":
            return AnalysisDepth.QUICK
        elif document_type == "scientific_research_paper":
            return AnalysisDepth.DEEP
        
        # Default to standard for other types
        return AnalysisDepth.STANDARD

    def get_config(self, depth: AnalysisDepth) -> DepthConfig:
        """Get configuration for specified depth."""
        return self.depth_configs[depth]

# Initialize global depth manager
depth_manager = AdaptiveDepthManager()