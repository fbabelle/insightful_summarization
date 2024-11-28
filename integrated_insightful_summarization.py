import textwrap
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from functools import lru_cache
import tiktoken
from dataclasses import dataclass
from __init__ import model_selection, MAX_TOKENS
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pdb, traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StitchingContext:
    section_type: str
    coherence_score: float
    key_entities: List[str]
    themes: List[str]
    transitions: List[str]

class SummaryStitcher:
    def __init__(self, model_name: str = 'avsolatorio/NoInstruct-small-Embedding-v0'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for text segments."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def _compute_coherence_score(self, seg1: str, seg2: str) -> float:
        """Compute semantic coherence between segments."""
        embeddings = self._get_embeddings([seg1, seg2])
        similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        return float(similarity.cpu())

    def _extract_context(self, segment: str, model: str = "gpt-4o") -> StitchingContext:
        """Extract contextual information from a segment."""
        prompt = f"""
        Analyze this text segment and extract:
        1. The type of section (introduction, body, conclusion, etc.)
        2. Key entities (names, concepts, etc.)
        3. Main themes
        4. Transition phrases or ideas

        Text: {segment}

        Respond in JSON format:
        {{
            "section_type": "type",
            "key_entities": ["entity1", "entity2"],
            "themes": ["theme1", "theme2"],
            "transitions": ["transition1", "transition2"]
        }}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = model_selection(model, messages=messages, output_json=True)
            result = json.loads(response)
            
            return StitchingContext(
                section_type=result["section_type"],
                coherence_score=0.0,  # Will be updated later
                key_entities=result["key_entities"],
                themes=result["themes"],
                transitions=result["transitions"]
            )
        except Exception as e:
            logger.error(f"Error in context extraction: {str(e)}")
            return StitchingContext("unknown", 0.0, [], [], [])

    def stitch_summaries(self, segments: List[str], model: str = "gpt-4o") -> str:
        """Stitch multiple summary segments into a coherent whole."""
        if not segments:
            return ""
        if len(segments) == 1:
            return segments[0]

        # Extract context for each segment
        contexts = [self._extract_context(seg) for seg in segments]
        
        # Compute coherence scores between adjacent segments
        for i in range(len(contexts) - 1):
            contexts[i].coherence_score = self._compute_coherence_score(
                segments[i], segments[i + 1]
            )

        # Generate transitions where needed
        stitched_segments = []
        for i in range(len(segments)):
            current_segment = segments[i]
            
            if i > 0:
                # Need to stitch with previous segment
                transition = self._generate_transition(
                    contexts[i-1], 
                    contexts[i],
                    segments[i-1],
                    current_segment,
                    model
                )
                current_segment = transition + current_segment

            stitched_segments.append(current_segment)

        # Final coherence check and refinement
        final_text = " ".join(stitched_segments)
        final_text = self._refine_stitched_text(final_text, contexts, model)
        
        return final_text

    def _generate_transition(
        self,
        context1: StitchingContext,
        context2: StitchingContext,
        segment1: str,
        segment2: str,
        model: str
    ) -> str:
        """Generate a transition between two segments."""
        prompt = f"""
        Generate a brief, natural transition between these segments.
        Consider their themes, entities, and logical flow.

        Segment 1 Type: {context1.section_type}
        Segment 1 Themes: {', '.join(context1.themes)}
        Segment 1 Key Points: {segment1[:200]}...

        Segment 2 Type: {context2.section_type}
        Segment 2 Themes: {', '.join(context2.themes)}
        Segment 2 Key Points: {segment2[:200]}...

        Generate a concise transition (1-2 sentences):
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            transition = model_selection(model, messages=messages, max_tokens=50)
            return f" {transition.strip()} "
        except Exception as e:
            logger.error(f"Error generating transition: {str(e)}")
            return " "

    def _refine_stitched_text(
        self,
        text: str,
        contexts: List[StitchingContext],
        model: str
    ) -> str:
        """Final refinement of the stitched text."""
        themes = set()
        entities = set()
        for ctx in contexts:
            themes.update(ctx.themes)
            entities.update(ctx.key_entities)

        prompt = f"""
        Refine this text to ensure coherence and flow.
        Maintain consistency in discussing these themes: {', '.join(themes)}
        Ensure proper handling of these key entities: {', '.join(entities)}

        Text: {text}

        Provide a refined version that maintains the same information but improves:
        1. Transition smoothness
        2. Theme consistency
        3. Logical flow
        4. Entity references

        Refined text:
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            return model_selection(model, messages=messages, max_tokens=2000)
        except Exception as e:
            logger.error(f"Error in final refinement: {str(e)}")
            return text


@dataclass
class AttentionLevel:
    name: str
    granularity: str
    weight: float
    focus_areas: List[str]

class MultiLevelAttention:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.attention_levels = {
            "global": AttentionLevel(
                name="global",
                granularity="document",
                weight=0.3,
                focus_areas=["main_thesis", "overall_structure", "key_conclusions"]
            ),
            "section": AttentionLevel(
                name="section",
                granularity="section",
                weight=0.3,
                focus_areas=["section_themes", "transitions", "sub_arguments"]
            ),
            "detail": AttentionLevel(
                name="detail",
                granularity="paragraph",
                weight=0.4,
                focus_areas=["specific_evidence", "technical_details", "citations"]
            )
        }

    def process_with_attention(
        self,
        content: str,
        analysis: Dict[str, Any],
        context: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Process content using multi-level attention mechanism.
        """
        attention_results = {}
        
        # Global level attention
        attention_results["global"] = self._apply_global_attention(content, analysis)
        
        # Section level attention
        attention_results["section"] = self._apply_section_attention(content, analysis)
        
        # Detail level attention
        attention_results["detail"] = self._apply_detail_attention(content, analysis, context)
        
        return self._integrate_attention_levels(attention_results)

    def _apply_global_attention(self, content: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply global level attention to capture overall document structure and main points.
        """
        prompt = f"""
        Analyze the following document at a global level, focusing on:
        1. Main thesis or central argument
        2. Overall document structure
        3. Key conclusions and their significance
        4. Broad research context

        Content: {content}

        Analysis Context:
        {json.dumps(analysis, indent=2)}

        Provide a JSON response with the following structure:
        {{
            "main_thesis": "overall thesis statement",
            "document_structure": ["major sections or components"],
            "key_conclusions": ["main conclusions"],
            "research_context": "broad context and significance"
        }}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = model_selection(self.model, messages=messages, output_json=True)
            return json.loads(response)
        except Exception as e:
            logging.error(f"Error in global attention: {str(e)}")
            return {}

    def _apply_section_attention(self, content: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply section level attention to capture thematic elements and transitions.
        """
        prompt = f"""
        Analyze the following document at a section level, focusing on:
        1. Major themes within each section
        2. Transitions between sections
        3. Supporting arguments and evidence
        4. Section-specific context

        Content: {content}

        Analysis Context:
        {json.dumps(analysis, indent=2)}

        Provide a JSON response with the following structure:
        {{
            "section_themes": [
                {{"section": "name", "theme": "description", "key_points": ["points"]}}
            ],
            "transitions": ["transition descriptions"],
            "supporting_arguments": ["main supporting points"],
            "sectional_context": ["relevant context for each section"]
        }}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = model_selection(self.model, messages=messages, output_json=True)
            return json.loads(response)
        except Exception as e:
            logging.error(f"Error in section attention: {str(e)}")
            return {}

    def _apply_detail_attention(
        self,
        content: str,
        analysis: Dict[str, Any],
        context: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Apply detail level attention to capture specific evidence and technical details.
        """
        context_str = "\n".join([f"Context {i+1}: {title}\n{text}" 
                                for i, (title, text) in enumerate(context)])
        
        prompt = f"""
        Analyze the following document at a detailed level, focusing on:
        1. Specific evidence and examples
        2. Technical details and methodology
        3. Citations and references
        4. Detailed context integration

        Content: {content}

        Analysis Context:
        {json.dumps(analysis, indent=2)}

        Related Context:
        {context_str}

        Provide a JSON response with the following structure:
        {{
            "key_evidence": ["specific evidence points"],
            "technical_details": ["important technical information"],
            "citations": ["relevant citations and their significance"],
            "detailed_context": ["specific contextual connections"]
        }}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = model_selection(self.model, messages=messages, output_json=True)
            return json.loads(response)
        except Exception as e:
            logging.error(f"Error in detail attention: {str(e)}")
            return {}

    def _integrate_attention_levels(self, attention_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate results from different attention levels into a coherent structure.
        """
        integrated = {
            "global_narrative": self._extract_global_narrative(attention_results["global"]),
            "thematic_elements": self._extract_thematic_elements(attention_results["section"]),
            "supporting_details": self._extract_supporting_details(attention_results["detail"])
        }
        
        # Calculate attention weights
        weights = {level.name: level.weight for level in self.attention_levels.values()}
        
        return {
            "integrated_analysis": integrated,
            "attention_weights": weights,
            "attention_distribution": self._calculate_attention_distribution(attention_results)
        }

    def _extract_global_narrative(self, global_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure global narrative elements."""
        return {
            "thesis": global_results.get("main_thesis", ""),
            "structure": global_results.get("document_structure", []),
            "conclusions": global_results.get("key_conclusions", []),
            "context": global_results.get("research_context", "")
        }

    def _extract_thematic_elements(self, section_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure thematic elements."""
        return {
            "themes": section_results.get("section_themes", []),
            "transitions": section_results.get("transitions", []),
            "arguments": section_results.get("supporting_arguments", [])
        }

    def _extract_supporting_details(self, detail_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure supporting details."""
        return {
            "evidence": detail_results.get("key_evidence", []),
            "technical_info": detail_results.get("technical_details", []),
            "citations": detail_results.get("citations", [])
        }

    def _calculate_attention_distribution(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the distribution of attention across different levels."""
        total_elements = {
            "global": len(self._flatten_dict(results["global"])),
            "section": len(self._flatten_dict(results["section"])),
            "detail": len(self._flatten_dict(results["detail"]))
        }
        
        total = sum(total_elements.values())
        if total == 0:
            return {k: v.weight for k, v in self.attention_levels.items()}
            
        return {
            level: count / total
            for level, count in total_elements.items()
        }

    def _flatten_dict(self, d: Dict[str, Any]) -> List[Any]:
        """Flatten a dictionary into a list of values."""
        flattened = []
        for v in d.values():
            if isinstance(v, list):
                flattened.extend(v)
            elif isinstance(v, dict):
                flattened.extend(self._flatten_dict(v))
            else:
                flattened.append(v)
        return flattened

def enhance_summary_generation(
    generate_insightful_summary_with_refinement: callable
) -> callable:
    """
    Enhance the existing summary generation function with multi-level attention.
    """
    def enhanced_function(
        title: str,
        basic_summary: str,
        analysis: Dict,
        domain: str,
        context: List[Tuple[str, str]],
        user_content: str,
        model: str = "gpt-4o"
    ) -> Optional[str]:
        # Initialize multi-level attention
        attention = MultiLevelAttention(model)
        
        # Process content with attention mechanism
        attention_results = attention.process_with_attention(user_content, analysis, context)
        
        # Add attention results to analysis
        analysis["attention_analysis"] = attention_results
        
        # Generate summary using enhanced analysis
        return generate_insightful_summary_with_refinement(
            title, basic_summary, analysis, domain, context, user_content, model
        )
    
    return enhanced_function


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

def process_context(context: List[Tuple[str, str]], user_title: str, user_content: str, model: str = "gpt-4o") -> str:
    """
    Process and summarize the context to extract core information related to the user's document.
    """
    context_prompt = f"""
    Analyze the following retrieved documents in relation to the user's document. 
    Extract and summarize the most relevant and core information that provides historical 
    and background context to the user's document. Focus on information that will help 
    create an insightful summary of the user's document.

    User's Document Title: {user_title}
    User's Document Content (excerpt): {truncate_to_token_limit(user_content, 2000, model)}

    Retrieved Documents:
    """
    
    for title, content in context:
        context_prompt += f"\nTitle: {title}\nContent (excerpt): {truncate_to_token_limit(content, 2000, model)}\n"

    context_prompt += """
    Provide a comprehensive summary of the core information from these documents that relates to and provides context for the user's document. 
    Focus on:
    1. Historical background
    2. Related research or events
    3. Key concepts or theories mentioned in multiple documents
    4. Contrasting viewpoints or debates in the field
    5. Recent developments or trends relevant to the user's document

    Your summary should be explicit and verbose, providing rich context for the user's document.
    """

    try:
        messages = [
            {"role": "system", "content": "You are an expert at extracting and summarizing core information from multiple sources."},
            {"role": "user", "content": context_prompt}
        ]

        core_info = model_selection(model, messages=messages, max_tokens=4000, temperature=0.3)
        token_count = count_tokens(core_info, model)
        logger.info(f"Core information extracted successfully. Length: {token_count} tokens")
        return core_info.strip()

    except Exception as e:
        logger.error(f"An error occurred while processing context: {str(e)}")
        return ""

def align_keys(analysis_keys: List[str], format_keys: List[str], model: str = "gpt-4o") -> Dict[str, str]:
    """
    Use LLM to align keys from the analysis dictionary with keys from the format_dict.
    
    Args:
    analysis_keys (List[str]): List of keys from the analysis dictionary
    format_keys (List[str]): List of keys from the format_dict
    model (str): The LLM model to use
    
    Returns:
    Dict[str, str]: A dictionary mapping analysis keys to format keys
    """
    prompt = f"""
    You are an expert in natural language understanding and semantic similarity. Your task is to align two sets of keys based on their semantic meaning and likely content. Some keys may not have a match, and that's okay.

    Set 1 (Analysis Keys): {', '.join(analysis_keys)}
    Set 2 (Format Keys): {', '.join(format_keys)}

    Please provide a JSON object where the keys are from Set 1, and the values are the most semantically similar keys from Set 2. If there's no good match, use null as the value.

    Example output format:
    {{
        "key_from_set1": "matching_key_from_set2",
        "another_key_from_set1": null
    }}

    Ensure your response is a valid JSON object.
    """

    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant skilled in understanding semantic similarities between words and phrases."},
            {"role": "user", "content": prompt}
        ]

        response = model_selection(model, messages=messages, temperature=0.3, output_json=True)
        alignment = json.loads(response)
        return alignment

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from the model")
        return {}
    except Exception as e:
        logger.error(f"An error occurred while aligning keys: {str(e)}")
        return {}

def construct_prompt(basic_summary: str, analysis: Dict, domain: str, core_context: str, model: str = "gpt-4o") -> str:
    """
    Constructs a prompt for the LLM to generate an insightful summary.
    """
    prompts = {
        'scientific_research_paper': textwrap.dedent("""
            You are an expert scientific communicator. Create an insightful summary of a scientific paper using the following information:

            Basic Summary: {basic_summary}

            Key Contributions: {contributions}
            Relation to Previous Work: {previous_work}
            Potential Impact: {impact}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly states the main findings and their significance
            2. Places the research in the context of the field, using the provided background information
            3. Explains the potential impact and applications of the work
            4. Identifies any limitations or areas for future research

            Your summary should be informative to both experts and informed lay readers.
        """),
        'news': textwrap.dedent("""
            You are an experienced journalist and analyst. Create an insightful summary of a news article using the following information:

            Basic Summary: {basic_summary}

            Key Events: {key_events}
            Broader Context: {context}
            Potential Implications: {implications}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly outlines the key events and their immediate significance
            2. Places the news in a broader context (historical, social, political, etc.) using the provided background information
            3. Explains why this is newsworthy and its potential impact
            4. Presents any relevant controversies or differing viewpoints

            Your summary should be informative and provide deeper insights than a typical news report.
        """),
        'article': textwrap.dedent("""
            You are a skilled literary analyst and critic. Create an insightful summary of an article or opinion piece using the following information:

            Basic Summary: {basic_summary}

            Main Argument: {main_argument}
            Author's Stance: {stance}
            Rhetorical Strategies: {rhetorical_strategies}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly states the main argument or thesis of the article
            2. Explains the author's stance and perspective
            3. Analyzes the rhetorical strategies and persuasive techniques used
            4. Evaluates the effectiveness of the argument and its potential impact
            5. Places the article in the broader context of the topic or debate, using the provided background information

            Your summary should provide a deeper understanding of both the content and the craft of the article.
        """)
    }
    
    prompt_template = prompts.get(domain, prompts['article'])
    
    # Create a dictionary with default values for all possible placeholders
    format_dict = {
        'basic_summary': basic_summary,
        'core_context': core_context,
        'contributions': 'Not specified',
        'previous_work': 'Not specified',
        'impact': 'Not specified',
        'key_events': 'Not specified',
        'context': 'Not specified',
        'implications': 'Not specified',
        'main_argument': 'Not specified',
        'stance': 'Not specified',
        'rhetorical_strategies': 'Not specified'
    }
    
    # Align keys between analysis and format_dict
    alignment = align_keys(list(analysis.keys()), list(format_dict.keys()), model)
    
    # Update format_dict with aligned values from analysis
    for analysis_key, format_key in alignment.items():
        if format_key and analysis_key in analysis:
            format_dict[format_key] = analysis[analysis_key]
    
    # Use the updated format_dict to fill in the prompt template
    return prompt_template.format(**format_dict)

def generate_insightful_summary(prompt: str, model: str = "gpt-4o") -> Optional[str]:
    """
    Generates an insightful summary using the specified LLM.
    """
    try:
        messages = [
            {"role": "system", "content": "You are an expert summarizer capable of providing insightful, well-structured summaries."},
            {"role": "user", "content": prompt}
        ]

        summary = model_selection(model, messages=messages, max_tokens=2000, temperature=0.3)
        token_count = count_tokens(summary, model)
        logger.info(f"Summary generated successfully. Length: {token_count} tokens")
        return summary.strip()

    except Exception as e:
        logger.error(f"An error occurred while generating the summary: {str(e)}")
        return None

def generate_insightful_summary_with_refinement(
    title: str,
    basic_summary: str,
    analysis: Dict,
    domain: str,
    context: List[Tuple[str, str]],
    user_content: str,
    model: str = "gpt-4o"
) -> Optional[str]:
    """Enhanced version with summary stitching."""
    logger.info("Starting insightful summary generation with stitching")
    
    # Initialize stitcher
    stitcher = SummaryStitcher()
    
    # Process parallel summaries
    parallel_summaries = []
    
    try:
        # Generate core summaries in parallel
        core_context = process_context(context, title, user_content, model)
        prompt = construct_prompt(basic_summary, analysis, domain, core_context)
        
        # Split into sections for parallel processing
        sections = [
            {"type": "overview", "content": basic_summary},
            {"type": "analysis", "content": json.dumps(analysis)},
            {"type": "context", "content": core_context}
        ]
        
        for section in sections:
            section_summary = generate_insightful_summary(
                f"Summarize this {section['type']} section:\n{section['content']}",
                model
            )
            if section_summary:
                parallel_summaries.append(section_summary)
        
        # Stitch the summaries together
        if parallel_summaries:
            final_summary = stitcher.stitch_summaries(parallel_summaries, model)
            logger.info("Summary stitching completed successfully")
            return final_summary
        
    except Exception as e:
        logger.error(f"Error in summary generation with stitching: {str(e)}")
    
    return None

def main():
    # Example usage
    title = "Climate Change Impacts on Global Biodiversity"
    basic_summary = "The paper discusses the effects of climate change on global biodiversity patterns."
    analysis = {
        "key_findings": "1. New climate-biodiversity model. 2. Global biodiversity impact assessment.",
        "literature_review": "Builds on IPCC reports and recent ecological studies.",
        "significance": "Potential for improved conservation strategies and policy making."
    }
    domain = "scientific_research_paper"
    context = [
        ("Recent Trends in Global Biodiversity Loss", "This paper examines the accelerating rate of species extinction..."),
        ("Climate Change: A Comprehensive Review", "An overview of climate change causes, effects, and mitigation strategies..."),
        ("Conservation Strategies in the Anthropocene", "Discussion of novel approaches to biodiversity conservation in the face of rapid global change...")
    ]
    user_content = "This study presents a comprehensive analysis of climate change impacts on global biodiversity..."

    enhanced_summarize = enhance_summary_generation(generate_insightful_summary_with_refinement)
    final_summary = enhanced_summarize(
        title, basic_summary, analysis, domain, context, user_content
    )
    # pdb.set_trace()
    if final_summary:
        logger.info("Final Insightful Summary:")
        logger.info(final_summary)
        logger.info(f"Final summary token count: {count_tokens(final_summary, 'gpt-4o')}")
    else:
        logger.error("Failed to generate insightful summary")

if __name__ == "__main__":
    main()