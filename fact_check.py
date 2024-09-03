import logging
from typing import List, Dict
from __init__ import model_selection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTrustedKnowledgeBase:
    def __init__(self, facts: List[str]):
        self.facts = facts

    def retrieve_relevant_facts(self, query: str, top_k: int = 5) -> List[str]:
        # Simple word matching
        query_words = set(query.lower().split())
        relevant_facts = []
        for fact in self.facts:
            fact_words = set(fact.lower().split())
            if query_words.intersection(fact_words):
                relevant_facts.append(fact)
            if len(relevant_facts) >= top_k:
                break
        return relevant_facts

def extract_claims(summary: str) -> List[str]:
    prompt = f"""
    Extract the key factual claims from the following summary. 
    Present each claim as a separate statement:

    Summary:
    {summary}

    Key Claims:
    1.
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with extracting key factual claims from summaries."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection("gpt-4o", messages=messages)
        claims = response.strip().split('\n')
        return [claim.split('. ', 1)[1] if '. ' in claim else claim for claim in claims if claim.strip()]
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        return []

def verify_claim(claim: str, relevant_facts: List[str]) -> str:
    prompt = f"""
    Verify the following claim against the provided facts:

    Claim: {claim}

    Relevant Facts:
    {' '.join(relevant_facts)}

    Is the claim supported by the facts? Respond with one of the following:
    - SUPPORTED: If the claim is directly supported by the facts.
    - PARTIALLY SUPPORTED: If parts of the claim are supported, but some details are missing or unclear.
    - NOT SUPPORTED: If the claim contradicts the facts or there's not enough information to support it.
    - UNCERTAIN: If there's not enough information to make a judgment.

    Explain your reasoning briefly.

    Verdict:
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with verifying claims against known facts."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = model_selection("gpt-4o", messages=messages)
        return response.strip()
    except Exception as e:
        logger.error(f"Error verifying claim: {str(e)}")
        return "UNCERTAIN: Unable to verify due to an error."

def fact_check(summary: str, trusted_knowledge_base: SimpleTrustedKnowledgeBase) -> str:
    claims = extract_claims(summary)
    verified_claims = []

    for claim in claims:
        relevant_facts = trusted_knowledge_base.retrieve_relevant_facts(claim)
        verification_result = verify_claim(claim, relevant_facts)
        verified_claims.append((claim, verification_result))

    verification_results = "\n".join([f"Claim: {claim}\nVerdict: {verdict}" for claim, verdict in verified_claims])

    fact_check_prompt = f"""
    Original summary:
    {summary}

    Verification results:
    {verification_results}

    Please rewrite the summary, taking into account the verification results. 
    Clearly indicate any claims that were not fully supported or were uncertain. 
    Maintain the overall structure and insights of the original summary where possible.

    Fact-checked summary:
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with generating fact-checked summaries."},
        {"role": "user", "content": fact_check_prompt}
    ]

    try:
        response = model_selection("gpt-4o", messages=messages)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating fact-checked summary: {str(e)}")
        return "Unable to generate fact-checked summary due to an error."

def main():
    # This is a mock trusted knowledge base. In a real scenario, this would be much larger and more comprehensive.
    mock_facts = [
        "The Earth orbits the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Great Wall of China is visible from space.",
        "The human body has 206 bones.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
    ]
    
    trusted_kb = SimpleTrustedKnowledgeBase(mock_facts)
    
    sample_summary = """
    The Earth, our home planet, follows an elliptical orbit around the Sun. 
    This journey takes approximately 365.25 days to complete, which is why we have leap years. 
    Interestingly, the Great Wall of China, one of the most impressive man-made structures, 
    is so large that it can be seen from the Moon with the naked eye. 
    The human body, a marvel of nature, consists of exactly 206 bones, which form our skeletal structure.
    """
    
    fact_checked_summary = fact_check(sample_summary, trusted_kb)
    logger.info("Fact-checked summary:\n%s", fact_checked_summary)

if __name__ == "__main__":
    main()