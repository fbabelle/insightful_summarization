import json
from typing import Dict, Any, List
from enum import Enum
import logging
from __init__ import model_selection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    SCIENTIFIC = 'scientific'
    NEWS = 'news'
    ARTICLE = 'article'

PROMPTS: Dict[AnalysisType, str] = {
    AnalysisType.SCIENTIFIC: """
    Analyze the following scientific paper and provide:
    1. Main contributions or findings (list up to 3)
    2. Key methods used (list up to 3)
    3. Significant results (list up to 3)
    4. Historical context: How does this research build upon or relate to previous breakthroughs in the field?
    5. Potential implications or applications (list up to 2)
    6. Key technical terms or concepts (list up to 5)

    Text to analyze:
    {text}

    Provide the analysis in JSON format with the following keys: 
    contributions, methods, results, previous_work, impact, key_terms
    """,
    AnalysisType.NEWS: """
    Analyze the following news article and provide:
    1. Main event or subject (1 sentence)
    2. Key entities involved (list up to 5: people, organizations, locations)
    3. Why is this event newsworthy? Explain its significance compared to regular events or observations.
    4. Potential impacts or consequences (list up to 3)
    5. Time relevance or urgency (1 sentence)
    6. Key phrases or buzzwords (list up to 5)

    Text to analyze:
    {text}

    Provide the analysis in JSON format with the following keys: 
    main_event, key_entities, newsworthiness, implications, time_relevance, key_phrases
    """,
    AnalysisType.ARTICLE: """
    Analyze the following article and provide:
    1. Main thesis or argument (1-2 sentences)
    2. Key supporting points (list up to 3)
    3. Author's perspective or bias (1 sentence)
    4. Identify the type of message (e.g., warning, appeal, informative, persuasive) and explain why
    5. Calls to action, if any (list up to 2)
    6. Key phrases or concepts (list up to 5)

    Text to analyze:
    {text}

    Provide the analysis in JSON format with the following keys: 
    main_argument, supporting_points, author_stance, message_type, calls_to_action, key_phrases
    """
}

def analyze_with_llm(text: str, analysis_type: AnalysisType) -> Dict[str, Any]:
    """
    Analyze the given text using GPT-4 based on the specified analysis type.
    
    Args:
    text (str): The text to be analyzed
    analysis_type (AnalysisType): The type of analysis to perform
    
    Returns:
    Dict[str, Any]: A dictionary containing the analysis results
    """
    prompt = PROMPTS[analysis_type].format(text=text)
    messages = [
        {"role": "system", "content": "You are an expert analyst skilled in extracting key information from texts."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection("gpt-4", messages=messages, temperature=0.5)
        analysis = json.loads(response)
        return analysis
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {str(e)}")
        return {"error": "Invalid JSON response from the model"}
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
        return {"error": str(e)}

def ensure_required_keys(analysis: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Ensure all required keys are present in the analysis dictionary."""
    for key in required_keys:
        if key not in analysis:
            analysis[key] = "Not specified"
    return analysis

def analyze_scientific_contributions(text: str) -> Dict[str, Any]:
    """Analyze the scientific contributions of a research paper."""
    analysis = analyze_with_llm(text, AnalysisType.SCIENTIFIC)
    required_keys = ['contributions', 'methods', 'results', 'previous_work', 'impact', 'key_terms']
    return ensure_required_keys(analysis, required_keys)

def analyze_news_significance(text: str) -> Dict[str, Any]:
    """Analyze the significance of a news article."""
    analysis = analyze_with_llm(text, AnalysisType.NEWS)
    required_keys = ['main_event', 'key_entities', 'newsworthiness', 'implications', 'time_relevance', 'key_phrases']
    return ensure_required_keys(analysis, required_keys)

def analyze_author_intent(text: str) -> Dict[str, Any]:
    """Analyze the author's intent in a general article."""
    analysis = analyze_with_llm(text, AnalysisType.ARTICLE)
    required_keys = ['main_argument', 'supporting_points', 'author_stance', 'message_type', 'calls_to_action', 'key_phrases']
    return ensure_required_keys(analysis, required_keys)

def main():
    # Example scientific paper text
    scientific_text = """
    In this paper, we introduce a novel method for quantum computing that significantly improves the efficiency of qubit operations. 
    Our approach, based on topological quantum circuits, demonstrates a 50% reduction in decoherence compared to traditional methods. 
    The results show promising applications in quantum error correction and may pave the way for more stable quantum computers.
    """
    logger.info("Scientific Paper Analysis:")
    logger.info(json.dumps(analyze_scientific_contributions(scientific_text), indent=2))
    
    # Example news article text
    news_text = """
    Breaking news: A major earthquake struck the coast of California today, measuring 7.2 on the Richter scale. 
    The impact was felt as far as 100 miles inland, affecting millions of residents. 
    Governor Jane Doe has declared a state of emergency and FEMA is mobilizing resources to assist in the aftermath.
    """
    logger.info("\nNews Article Analysis:")
    logger.info(json.dumps(analyze_news_significance(news_text), indent=2))
    
    # Example general article text
    article_text = """
    The rise of artificial intelligence is reshaping our world in unprecedented ways. 
    This article argues that while AI offers immense benefits, we must also address the ethical implications and potential job displacement. 
    It is crucial that policymakers and tech leaders work together to ensure AI development benefits all of society. 
    We need to act now to shape the future of AI before it's too late.
    """
    logger.info("\nGeneral Article Analysis:")
    logger.info(json.dumps(analyze_author_intent(article_text), indent=2))

if __name__ == "__main__":
    main()