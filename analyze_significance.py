import os
import json
from typing import Dict, Any, List
import logging
from functools import partial, lru_cache
import multiprocessing
import tiktoken
from enum import Enum
from dataclasses import dataclass
from __init__ import model_selection, MAX_TOKENS

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    SCIENTIFIC_RESEARCH = "scientific_research_paper"
    NEWS_ARTICLE = "news"
    GENERAL_ARTICLE = "article"

@dataclass
class SignificanceCriteria:
    description: str
    evaluation_points: List[str]
    weight: float

class SignificanceScorer:
    def __init__(self):
        self.criteria_weights = {
            DocumentType.SCIENTIFIC_RESEARCH: {
                "novelty": 0.35,
                "impact": 0.25,
                "methodology": 0.25,
                "citation_network": 0.15
            },
            DocumentType.NEWS_ARTICLE: {
                "novelty": 0.20,
                "impact": 0.40,
                "methodology": 0.15,
                "citation_network": 0.25
            },
            DocumentType.GENERAL_ARTICLE: {
                "novelty": 0.25,
                "impact": 0.35,
                "methodology": 0.20,
                "citation_network": 0.20
            }
        }

    def calculate_significance(self, doc_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate significance scores based on document analysis"""
        doc_type_enum = DocumentType(doc_type)
        weights = self.criteria_weights[doc_type_enum]
        scores = {}
        
        if doc_type == "scientific_research_paper":
            scores = {
                "novelty": self._score_novelty_research(analysis),
                "impact": self._score_impact_research(analysis),
                "methodology": self._score_methodology_research(analysis),
                "citation_network": self._score_citation_research(analysis)
            }
        elif doc_type == "news":
            scores = {
                "novelty": self._score_novelty_news(analysis),
                "impact": self._score_impact_news(analysis),
                "methodology": self._score_methodology_news(analysis),
                "citation_network": self._score_citation_news(analysis)
            }
        else:  # article
            scores = {
                "novelty": self._score_novelty_article(analysis),
                "impact": self._score_impact_article(analysis),
                "methodology": self._score_methodology_article(analysis),
                "citation_network": self._score_citation_article(analysis)
            }

        weighted_scores = {
            criteria: score * weights[criteria]
            for criteria, score in scores.items()
        }

        return {
            "raw_scores": scores,
            "weighted_scores": weighted_scores,
            "total_significance": sum(weighted_scores.values())
        }

    def _score_novelty_research(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'contributions' in analysis:
            score += min(len(analysis['contributions']) * 0.2, 0.6)
        if 'previous_work' in analysis:
            score += 0.4 if analysis['previous_work'] else 0.0
        return min(score, 1.0)

    def _score_impact_research(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'impact' in analysis:
            score += min(len(analysis['impact']) * 0.3, 0.6)
        if 'results' in analysis:
            score += min(len(analysis['results']) * 0.2, 0.4)
        return min(score, 1.0)

    def _score_methodology_research(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'methods' in analysis:
            score += min(len(analysis['methods']) * 0.3, 0.9)
        if 'key_terms' in analysis:
            score += min(len(analysis['key_terms']) * 0.02, 0.1)
        return min(score, 1.0)

    def _score_citation_research(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'previous_work' in analysis:
            score += 0.6 if analysis['previous_work'] else 0.0
        if 'key_terms' in analysis:
            score += min(len(analysis['key_terms']) * 0.08, 0.4)
        return min(score, 1.0)

    # Similar scoring methods for news articles
    def _score_novelty_news(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'newsworthiness' in analysis:
            score += 0.6
        if 'time_relevance' in analysis:
            score += 0.4
        return min(score, 1.0)

    def _score_impact_news(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'implications' in analysis:
            score += min(len(analysis['implications']) * 0.3, 0.9)
        if 'key_entities' in analysis:
            score += min(len(analysis['key_entities']) * 0.02, 0.1)
        return min(score, 1.0)

    def _score_methodology_news(self, analysis: Dict[str, Any]) -> float:
        return 0.5  # Base score for news articles

    def _score_citation_news(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'key_entities' in analysis:
            score += min(len(analysis['key_entities']) * 0.2, 1.0)
        return min(score, 1.0)

    # Scoring methods for general articles
    def _score_novelty_article(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'main_argument' in analysis:
            score += 0.5
        if 'supporting_points' in analysis:
            score += min(len(analysis['supporting_points']) * 0.1, 0.5)
        return min(score, 1.0)

    def _score_impact_article(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'calls_to_action' in analysis:
            score += min(len(analysis['calls_to_action']) * 0.3, 0.6)
        if 'message_type' in analysis:
            score += 0.4
        return min(score, 1.0)

    def _score_methodology_article(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'supporting_points' in analysis:
            score += min(len(analysis['supporting_points']) * 0.2, 0.6)
        if 'key_phrases' in analysis:
            score += min(len(analysis['key_phrases']) * 0.08, 0.4)
        return min(score, 1.0)

    def _score_citation_article(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if 'key_phrases' in analysis:
            score += min(len(analysis['key_phrases']) * 0.2, 1.0)
        return min(score, 1.0)

# Update the analyze_document function to include significance scoring
def analyze_document(title: str, content: str, analysis_type: str) -> Dict[str, Any]:
    """
    Analyze the document and calculate its significance score.
    """
    if analysis_type not in PROMPTS:
        raise ValueError(f"Invalid analysis type: {analysis_type}")

    chunks = chunk_text(content)
    
    with multiprocessing.Pool() as pool:
        chunk_analyses = pool.map(partial(analyze_chunk, analysis_type=analysis_type), chunks)
    
    merged_analysis = merge_analyses(chunk_analyses, analysis_type)
    merged_analysis['title'] = title
    
    # Calculate significance score
    scorer = SignificanceScorer()
    doc_type = DocumentType(analysis_type)
    significance_scores = scorer.calculate_significance(doc_type, merged_analysis)
    
    # Add significance scores to the analysis
    merged_analysis['significance_analysis'] = significance_scores
    
    return merged_analysis



PROMPTS = {
    'scientific_research_paper': """
    Analyze the following scientific / research paper and provide:
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
    'news': """
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
    'article': """
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

@lru_cache(maxsize=128)
def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)

def chunk_text(text: str, max_tokens: int = MAX_TOKENS // 2, overlap: float = 0.1) -> List[str]:
    encoding = get_encoding()
    tokens = encoding.encode(text)
    chunks = []
    overlap_tokens = int(max_tokens * overlap)
    
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start = end - overlap_tokens
        start = max(start, end - max_tokens)
        if end == len(tokens):
            break

    return chunks

def analyze_chunk(chunk: str, analysis_type: str) -> Dict[str, Any]:
    if analysis_type not in PROMPTS:
        raise ValueError(f"Invalid analysis type: {analysis_type}")

    prompt = PROMPTS[analysis_type].format(text=chunk)
    messages = [
        {"role": "system", "content": "You are an expert analyst skilled in extracting key information from texts."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection("gpt-4o", messages=messages, output_json=True, temperature=0.1)
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {str(e)}")
        return {"error": "Invalid JSON response from the model"}
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
        return {"error": str(e)}

def merge_analyses(analyses: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
    """
    Merge multiple chunk analyses into a single comprehensive analysis.
    """
    merged = {}
    if analysis_type == 'scientific_research_paper':
        merged = {
            'contributions': [],
            'methods': [],
            'results': [],
            'previous_work': '',
            'impact': [],
            'key_terms': []
        }
        for analysis in analyses:
            merged['contributions'].extend(analysis.get('contributions', []))
            merged['methods'].extend(analysis.get('methods', []))
            merged['results'].extend(analysis.get('results', []))
            merged['previous_work'] += analysis.get('previous_work', '') + ' '
            merged['impact'].extend(analysis.get('impact', []))
            merged['key_terms'].extend(analysis.get('key_terms', []))
    elif analysis_type == 'news':
        merged = {
            'main_event': '',
            'key_entities': [],
            'newsworthiness': '',
            'implications': [],
            'time_relevance': '',
            'key_phrases': []
        }
        for analysis in analyses:
            merged['main_event'] += analysis.get('main_event', '') + ' '
            merged['key_entities'].extend(analysis.get('key_entities', []))
            merged['newsworthiness'] += analysis.get('newsworthiness', '') + ' '
            merged['implications'].extend(analysis.get('implications', []))
            merged['time_relevance'] += analysis.get('time_relevance', '') + ' '
            merged['key_phrases'].extend(analysis.get('key_phrases', []))
    elif analysis_type == 'article':
        merged = {
            'main_argument': '',
            'supporting_points': [],
            'author_stance': '',
            'message_type': '',
            'calls_to_action': [],
            'key_phrases': []
        }
        for analysis in analyses:
            merged['main_argument'] += analysis.get('main_argument', '') + ' '
            merged['supporting_points'].extend(analysis.get('supporting_points', []))
            merged['author_stance'] += analysis.get('author_stance', '') + ' '
            merged['message_type'] += analysis.get('message_type', '') + ' '
            merged['calls_to_action'].extend(analysis.get('calls_to_action', []))
            merged['key_phrases'].extend(analysis.get('key_phrases', []))

    # Remove duplicates and limit list lengths
    for key in merged:
        if isinstance(merged[key], list):
            merged[key] = list(dict.fromkeys(merged[key]))[:5]
        elif isinstance(merged[key], str):
            merged[key] = merged[key].strip()

    return merged

def analyze_document(title: str, content: str, analysis_type: str) -> Dict[str, Any]:
    """Analyze the document and calculate its significance score."""
    if analysis_type not in PROMPTS:
        raise ValueError(f"Invalid analysis type: {analysis_type}")

    # Analyze content
    chunks = chunk_text(content)
    with multiprocessing.Pool() as pool:
        chunk_analyses = pool.map(partial(analyze_chunk, analysis_type=analysis_type), chunks)
    
    merged_analysis = merge_analyses(chunk_analyses, analysis_type)
    merged_analysis['title'] = title

    # Calculate significance score
    scorer = SignificanceScorer()
    try:
        significance_scores = scorer.calculate_significance(analysis_type, merged_analysis)
        merged_analysis['significance_analysis'] = significance_scores
    except Exception as e:
        logger.error(f"Error calculating significance: {str(e)}")
        merged_analysis['significance_analysis'] = {"error": str(e)}

    return merged_analysis

def main():
    # Example documents
    documents = [
        {
            "title": "The Impact and Challenges of Climate Change",
            "content": """
            Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels.

            These activities release greenhouse gases into the atmosphere, primarily carbon dioxide and methane. These gases trap heat from the sun, causing the Earth's average temperature to rise. This phenomenon is known as the greenhouse effect.

            The impacts of climate change are far-reaching and significant. They include more frequent and severe weather events, such as hurricanes, droughts, and heatwaves. Rising sea levels threaten coastal communities and islands. Changes in temperature and precipitation patterns affect agriculture and food security.

            Moreover, climate change poses risks to human health, biodiversity, and economic stability. It exacerbates existing social and economic inequalities, as vulnerable populations often bear the brunt of its effects.

            Addressing climate change requires a multi-faceted approach. This includes transitioning to renewable energy sources, improving energy efficiency, protecting and restoring ecosystems, and adapting to the changes already set in motion. International cooperation, as seen in agreements like the Paris Accord, is crucial in coordinating global efforts to mitigate climate change.

            While the challenge is immense, there is still time to act. Every individual, business, and government has a role to play in reducing greenhouse gas emissions and building a sustainable future. The choices we make today will determine the world we leave for future generations.
            """,
            "type": "article"
        },
        {
            "title": "Breakthrough in Quantum Computing Achieves 100-Qubit Processor",
            "content": """
            In a groundbreaking development, researchers at QuantumTech Labs have successfully created and operated a 100-qubit quantum processor, marking a significant milestone in the field of quantum computing.

            The team, led by Dr. Emily Quantum, utilized a novel approach combining superconducting circuits and topological error correction to achieve unprecedented stability and coherence in their qubit system. This advancement addresses one of the key challenges in quantum computing: maintaining quantum states for extended periods.

            The 100-qubit processor demonstrated the ability to perform complex quantum algorithms that are beyond the reach of classical computers. In one test, it solved a optimization problem in minutes that would take a traditional supercomputer years to complete.

            This breakthrough builds upon previous work in the field, including Google's 53-qubit quantum supremacy claim in 2019. However, the QuantumTech Labs' processor represents a quantum leap in both scale and reliability.

            The implications of this achievement are far-reaching. Potential applications include accelerated drug discovery, enhanced cryptography, and more efficient solutions to complex logistical problems. The financial sector is particularly interested in its potential for portfolio optimization and risk assessment.

            As exciting as this development is, the researchers caution that practical, large-scale quantum computers are still years away. Challenges remain in scaling up the technology and in developing quantum-specific software and algorithms.

            Nevertheless, this breakthrough brings us one step closer to the quantum computing revolution, promising to transform industries and tackle some of humanity's most complex challenges.
            """,
            "type": "scientific_research_paper"
        }
    ]
    
    for doc in documents:
        logger.info(f"\nAnalyzing document: {doc['title']}")
        analysis_result = analyze_document(doc['title'], doc['content'], doc['type'])
        logger.info(f"Analysis Result for {doc['title']}:")
        logger.info(json.dumps(analysis_result, indent=2))

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Necessary for Windows compatibility
    main()