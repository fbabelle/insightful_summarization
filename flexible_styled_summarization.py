from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from integrated_insightful_summarization import generate_insightful_summary_with_refinement, MultiLevelAttention

class SummaryStyle(Enum):
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    GENERAL = "general"

@dataclass
class SummaryParameters:
    max_length: int
    style: SummaryStyle
    technical_level: str
    target_audience: str
    include_citations: bool = False
    include_figures: bool = False

def enhance_summary_generation_with_flexibility(
    generate_insightful_summary_with_refinement: callable
) -> callable:
    """
    Enhanced decorator that combines multi-level attention with flexible summarization.
    """
    def enhanced_function(
        title: str,
        basic_summary: str,
        analysis: Dict,
        domain: str,
        context: List[Tuple[str, str]],
        user_content: str,
        summary_params: Optional[SummaryParameters] = None,
        model: str = "gpt-4o"
    ) -> Optional[str]:
        # Initialize multi-level attention
        attention = MultiLevelAttention(model)
        
        # Process content with attention mechanism
        attention_results = attention.process_with_attention(user_content, analysis, context)
        
        # Add attention results to analysis
        analysis["attention_analysis"] = attention_results

        # Apply flexibility parameters if provided
        if summary_params:
            # Adjust prompts based on summary parameters
            modified_prompt = adjust_prompt_for_style(
                construct_prompt(basic_summary, analysis, domain, 
                               process_context(context, title, user_content, model)),
                summary_params
            )
            
            # Generate summary with adjusted parameters
            return generate_insightful_summary(
                modified_prompt,
                model,
                max_tokens=summary_params.max_length
            )
        else:
            # Use default summary generation
            return generate_insightful_summary_with_refinement(
                title, basic_summary, analysis, domain, context, user_content, model
            )
    
    return enhanced_function

def adjust_prompt_for_style(base_prompt: str, params: SummaryParameters) -> str:
    """
    Adjust the prompt based on the desired summary style and parameters.
    """
    style_adjustments = {
        SummaryStyle.TECHNICAL: {
            "prefix": "Generate a technical summary with detailed methodology and findings.",
            "terminology": "specialized",
            "focus": "technical details and methodological rigor"
        },
        SummaryStyle.EXECUTIVE: {
            "prefix": "Generate an executive summary focusing on key insights and implications.",
            "terminology": "business-oriented",
            "focus": "strategic implications and actionable insights"
        },
        SummaryStyle.GENERAL: {
            "prefix": "Generate an accessible summary for a general audience.",
            "terminology": "clear and accessible",
            "focus": "main concepts and practical implications"
        }
    }
    
    style = style_adjustments[params.style]
    
    adjusted_prompt = f"""
    {style['prefix']}

    Target Audience: {params.target_audience}
    Technical Level: {params.technical_level}
    Use {style['terminology']} terminology and focus on {style['focus']}.

    {base_prompt}

    Additional Requirements:
    - Maximum Length: {params.max_length} tokens
    {f'- Include relevant citations and references' if params.include_citations else ''}
    {f'- Reference key figures and diagrams' if params.include_figures else ''}
    """
    
    return adjusted_prompt

# Example usage:
def example_usage():
    # Define flexible summary parameters
    summary_params = SummaryParameters(
        max_length=1000,
        style=SummaryStyle.TECHNICAL,
        technical_level="advanced",
        target_audience="researchers",
        include_citations=True
    )
    
    # Create enhanced summarizer with flexibility
    enhanced_summarize = enhance_summary_generation_with_flexibility(
        generate_insightful_summary_with_refinement
    )
    
    # Example document
    title = "Climate Change Impacts on Global Biodiversity"
    basic_summary = "The paper discusses the effects of climate change on global biodiversity patterns."
    analysis = {
        "key_findings": "1. New climate-biodiversity model. 2. Global biodiversity impact assessment.",
        "literature_review": "Builds on IPCC reports and recent ecological studies.",
        "significance": "Potential for improved conservation strategies and policy making."
    }
    domain = "scientific_research_paper"
    context = [
        ("Recent Trends in Global Biodiversity Loss", 
         "This paper examines the accelerating rate of species extinction..."),
        ("Climate Change: A Comprehensive Review", 
         "An overview of climate change causes, effects, and mitigation strategies...")
    ]
    user_content = "This study presents a comprehensive analysis of climate change impacts..."
    
    # Generate summary with flexible parameters
    final_summary = enhanced_summarize(
        title=title,
        basic_summary=basic_summary,
        analysis=analysis,
        domain=domain,
        context=context,
        user_content=user_content,
        summary_params=summary_params
    )
    
    if final_summary:
        logger.info("Generated Flexible Summary:")
        logger.info(final_summary)
    else:
        logger.error("Failed to generate flexible summary")

if __name__ == "__main__":
    example_usage()