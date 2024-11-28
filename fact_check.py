import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import tiktoken
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from __init__ import model_selection, MAX_TOKENS, fact_check_cache, num_tokens_from_string
import pdb, traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    def __init__(self):
        # Common date patterns in text
        self.date_patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # 2024-01-15
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',  # January 15, 2024
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),\s+\d{4})',  # 15 January 2024
            r'in\s+(\d{4})',  # in 2024
            r'as\s+of\s+(\d{4})',  # as of 2024
            r'(\d{4})\s+(?:report|data|statistics|figures)'  # 2024 report/data
        ]
        
        # Time-sensitive keywords
        self.temporal_indicators = {
            'current': timedelta(days=30),
            'recent': timedelta(days=90),
            'latest': timedelta(days=30),
            'now': timedelta(days=7),
            'today': timedelta(days=1),
            'this year': timedelta(days=365),
            'this month': timedelta(days=30),
            'this week': timedelta(days=7)
        }

    def extract_date_from_text(self, text: str) -> Optional[str]:
        """Extract date information from text content."""
        import re
        from datetime import datetime

        # Try to find dates in text
        for pattern in self.date_patterns:
            matches = re.search(pattern, text)
            if matches:
                try:
                    # Try to parse the found date
                    date_str = matches.group(1)
                    # Handle different date formats
                    for fmt in ["%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y"]:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            return date.strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                except Exception:
                    continue

        return None

    def infer_temporal_context(self, text: str, extracted_date: Optional[str] = None) -> Dict[str, Any]:
        """Infer temporal context from content."""
        context = {
            "date": extracted_date,
            "temporal_type": "UNCERTAIN",
            "confidence": 0.5,
            "warning": None
        }

        # Check for historical content
        if any(word in text.lower() for word in ['historical', 'history', 'past', 'traditionally']):
            context.update({
                "temporal_type": "HISTORICAL",
                "confidence": 0.8,
                "warning": "Historical information - may still be relevant"
            })
            return context

        # Check for time-sensitive indicators
        for indicator, validity_period in self.temporal_indicators.items():
            if indicator in text.lower():
                context.update({
                    "temporal_type": "TIME_SENSITIVE",
                    "confidence": 0.7,
                    "warning": f"Time-sensitive information (typical validity: {validity_period.days} days)"
                })
                return context

        # If we have a date, assess its recency
        if extracted_date:
            try:
                date = datetime.strptime(extracted_date, "%Y-%m-%d")
                age = datetime.now() - date
                
                if age <= timedelta(days=90):
                    context.update({
                        "temporal_type": "CURRENT",
                        "confidence": 0.9,
                        "warning": None
                    })
                elif age <= timedelta(days=365):
                    context.update({
                        "temporal_type": "RECENT",
                        "confidence": 0.7,
                        "warning": "Information is from within the past year"
                    })
                else:
                    context.update({
                        "temporal_type": "OUTDATED",
                        "confidence": 0.6,
                        "warning": f"Information is {age.days // 365} years old"
                    })
            except Exception:
                pass

        return context

def enhance_context_with_temporal_info(context: List[Tuple[str, str]]) -> List[Tuple[str, str, Optional[str]]]:
    """Enhance existing context with temporal information."""
    analyzer = TemporalAnalyzer()
    enhanced_context = []

    for title, content in context:
        # Try to extract date from both title and content
        date = analyzer.extract_date_from_text(title) or analyzer.extract_date_from_text(content)
        
        # Get temporal context
        temporal_info = analyzer.infer_temporal_context(content, date)
        
        # Add temporal information to content
        enhanced_content = {
            "original_content": content,
            "temporal_context": temporal_info
        }
        
        enhanced_context.append((title, json.dumps(enhanced_content), date))

    return enhanced_context

class TemporalRelevance(Enum):
    CURRENT = "current"
    OUTDATED = "outdated"
    TIME_SENSITIVE = "sensitive"
    HISTORICAL = "historical"
    UNCERTAIN = "uncertain"

@dataclass
class TemporalContext:
    publication_date: Optional[datetime]
    valid_until: Optional[datetime]
    temporal_type: TemporalRelevance


@lru_cache(maxsize=128)
def get_encoding(model: str) -> tiktoken.Encoding:
    """Returns the encoding for the specified model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"No specific tokenizer found for {model}. Using cl100k_base as default.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in the given text for the specified model."""
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def truncate_to_token_limit(text: str, max_tokens: int, model: str) -> str:
    """Truncate the text to fit within the specified token limit."""
    encoding = get_encoding(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


class RetrievalBasedKnowledgeBase:
    def __init__(self, context: List[Tuple[str, str, Optional[str]]]):  # Modified to accept dates
        """
        Initialize the knowledge base with retrieval results and temporal information.
        
        Args:
        context (List[Tuple[str, str, str]]): List of (title, content, date) tuples
        """
        self.facts = self._process_retrieval_results(context)
        self._build_index()

    def _assess_temporal_relevance(self, content: str, date_str: Optional[str]) -> TemporalContext:
        """Assess temporal relevance of content."""
        try:
            pub_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
            current_date = datetime.now()

            # Define time-sensitive keywords and their validity periods
            time_indicators = {
                'current': timedelta(days=30),
                'recent': timedelta(days=90),
                'latest': timedelta(days=30),
                'now': timedelta(days=7),
                'today': timedelta(days=1)
            }

            # Check for historical content
            if any(word in content.lower() for word in ['historical', 'history', 'past']):
                return TemporalContext(pub_date, None, TemporalRelevance.HISTORICAL)

            # If no date provided
            if not pub_date:
                return TemporalContext(None, None, TemporalRelevance.UNCERTAIN)

            # Check for time-sensitive content
            for keyword, validity_period in time_indicators.items():
                if keyword in content.lower():
                    valid_until = pub_date + validity_period
                    if current_date > valid_until:
                        return TemporalContext(pub_date, valid_until, TemporalRelevance.OUTDATED)
                    return TemporalContext(pub_date, valid_until, TemporalRelevance.TIME_SENSITIVE)

            # Default case - check if content is too old (e.g., > 1 year)
            if (current_date - pub_date) > timedelta(days=365):
                return TemporalContext(pub_date, None, TemporalRelevance.OUTDATED)

            return TemporalContext(pub_date, None, TemporalRelevance.CURRENT)

        except Exception as e:
            logger.error(f"Error in temporal assessment: {str(e)}")
            return TemporalContext(None, None, TemporalRelevance.UNCERTAIN)

    def _process_retrieval_results(self, context: List[Tuple[str, str, Optional[str]]]) -> List[Dict[str, Any]]:
        """Process retrieval results with temporal awareness."""
        processed_facts = []

        for title, content, date in context:
            temporal_context = self._assess_temporal_relevance(content, date)
            keywords = self._extract_keywords(title + " " + content)
            
            processed_facts.append({
                "title": title,
                "statement": content,
                "temporal_info": {
                    "publication_date": date,
                    "temporal_type": temporal_context.temporal_type.value,
                    "valid_until": temporal_context.valid_until.isoformat() if temporal_context.valid_until else None
                },
                "keywords": keywords
            })
        return processed_facts

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from the text using a simple frequency-based approach."""
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'])
        words = [word for word in words if word not in stop_words]
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        return sorted(word_freq, key=word_freq.get, reverse=True)[:max_keywords]

    def _build_index(self):
        """Build an inverted index for faster retrieval."""
        self.keyword_index = defaultdict(list)
        for i, fact in enumerate(self.facts):
            for keyword in fact["keywords"]:
                self.keyword_index[keyword.lower()].append(i)

    def retrieve_relevant_facts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant facts based on the query."""
        query_words = set(query.lower().split())
        fact_scores = defaultdict(float)
        for word in query_words:
            if word in self.keyword_index:
                for fact_index in self.keyword_index[word]:
                    fact_scores[fact_index] += 1
        sorted_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.facts[fact_index] for fact_index, _ in sorted_facts[:top_k]]

def extract_claims(summary: str, model: str = "gpt-4o") -> List[str]:
    """Extract key factual claims from the summary."""
    prompt = f"""
    Extract the key factual claims from the following summary. 
    Present each claim as a separate statement:

    Summary:
    {truncate_to_token_limit(summary, 3000, model)}

    Key Claims:
    1.
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with extracting key factual claims from summaries."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection(model, messages=messages, max_tokens=1000)
        claims = response.strip().split('\n')
        return [claim.split('. ', 1)[1] if '. ' in claim else claim for claim in claims if claim.strip()]
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        return []

def verify_claim(claim: str, relevant_facts: List[Dict[str, Any]], model: str = "gpt-4o") -> Dict[str, Any]:
    """Verify a claim with temporal awareness."""
    prompt = f"""
    Verify the following claim against the provided facts, considering temporal context:

    Claim: {claim}

    Relevant Facts (with temporal information):
    {json.dumps(relevant_facts, indent=2)}

    Provide a JSON response with:
    {{
        "verdict": "SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED/UNCERTAIN",
        "temporal_status": "CURRENT/OUTDATED/TIME_SENSITIVE/HISTORICAL/UNCERTAIN",
        "explanation": "Detailed explanation including temporal considerations",
        "confidence": float,  # 0-1
        "temporal_warning": "Any warnings about time-sensitivity or currency"
    }}
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in temporal-aware fact verification."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection(model, messages=messages, output_json=True)
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error in temporal verification: {str(e)}")
        return {
            "verdict": "UNCERTAIN",
            "temporal_status": "UNCERTAIN",
            "explanation": f"Verification error: {str(e)}",
            "confidence": 0.0,
            "temporal_warning": "Unable to verify temporal context"
        }

def fact_check(summary: str, context: List[Tuple[str, str]], model: str = "gpt-4o") -> Optional[str]:
    """
    Perform fact-checking with temporal awareness, working with standard context format.
    """

    """Fact checking with caching."""
    cache_key = f"fact_check:{hash(summary + str(context))}"
    
    # Check cache first
    cached_result = fact_check_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Cache hit for fact checking")
        return cached_result

    # Enhance context with temporal information
    enhanced_context = enhance_context_with_temporal_info(context)
    
    # Create knowledge base with enhanced context
    knowledge_base = RetrievalBasedKnowledgeBase(enhanced_context)
    claims = extract_claims(summary, model)
    verified_claims = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_claim = {
            executor.submit(
                verify_claim,
                claim,
                knowledge_base.retrieve_relevant_facts(claim),
                model
            ): claim for claim in claims
        }

        for future in as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                result = future.result()
                verified_claims.append((claim, result))
            except Exception as e:
                logger.error(f"Error verifying claim '{claim}': {str(e)}")

    # Format verification results with temporal information
    verification_results = "\n".join([
        f"Claim: {claim}\n"
        f"Verdict: {result['verdict']}\n"
        f"Temporal Status: {result.get('temporal_status', 'UNCERTAIN')}\n"
        f"Confidence: {result['confidence']}\n"
        f"Explanation: {result['explanation']}\n"
        f"Temporal Warning: {result.get('temporal_warning', 'No temporal information available')}"
        for claim, result in verified_claims
    ])
    
    # Enhanced prompt for fact-checked summary generation
    fact_check_prompt = f"""
    Original summary:
    {truncate_to_token_limit(summary, 2000, model)}

    Verification results with temporal context:
    {truncate_to_token_limit(verification_results, 4000, model)}

    Please rewrite the summary, considering both factual accuracy and temporal relevance:
    1. Indicate any claims that were not fully supported or were uncertain
    2. Add temporal context where relevant (e.g., "as of [date]", "current as of", etc.)
    3. Flag any time-sensitive information
    4. Note when information might need updating
    5. Maintain the overall structure and insights where possible

    Now rewrite the summary with maximum retained information and expressiveness.

    Fact-checked summary:
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in generating temporally-aware fact-checked summaries."},
        {"role": "user", "content": fact_check_prompt}
    ]

    try:
        response = model_selection(model, messages=messages, max_tokens=2000)
        fact_checked_summary = response.strip()
        logger.info(f"Temporally-aware fact-checked summary generated. Token count: {count_tokens(fact_checked_summary, model)}")

        # Cache the result
        if fact_checked_summary:
            token_count = num_tokens_from_string(fact_checked_summary)
            fact_check_cache.put(cache_key, fact_checked_summary, token_count)

        return fact_checked_summary
    except Exception as e:
        logger.error(f"Error generating fact-checked summary: {str(e)}")
        return None

def main():
    # Example usage
    summary = """
    The Earth, our home planet, follows an elliptical orbit around the Sun. 
    This journey takes approximately 365.25 days to complete, which is why we have leap years. 
    Interestingly, the Great Wall of China, one of the most impressive man-made structures, 
    is so large that it can be seen from the Moon with the naked eye. 
    The human body, a marvel of nature, consists of exactly 206 bones, which form our skeletal structure.
    """

    context = [
        ("Earth's Orbit", "The Earth orbits the Sun in an elliptical path. This journey takes approximately 365.25 days to complete, which is why we have leap years."),
        ("Water Boiling Point", "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level. This temperature can vary based on atmospheric pressure."),
        ("Great Wall of China Visibility", "Contrary to popular belief, the Great Wall of China is not visible from space with the naked eye. This myth has been debunked by numerous astronauts."),
        ("Human Skeletal System", "The adult human body typically contains 206 bones. This number can vary slightly due to anatomical variations in some individuals."),
        ("Mona Lisa Painter", "The Mona Lisa, one of the most famous paintings in the world, was created by the Italian Renaissance artist Leonardo da Vinci. It is housed in the Louvre Museum in Paris.")
    ]

    fact_checked_summary = fact_check(summary, context)
    if fact_checked_summary:
        logger.info("Temporally-aware fact-checked summary:")
        logger.info(fact_checked_summary)
        logger.info(f"Token count of fact-checked summary: {count_tokens(fact_checked_summary, 'gpt-4o')}")
    else:
        logger.error("Failed to generate fact-checked summary")

if __name__ == "__main__":
    main()