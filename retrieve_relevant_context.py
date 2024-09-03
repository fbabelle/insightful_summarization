import pickle
import os
import logging
from typing import List, Dict
from collections import Counter
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleContextRetriever:
    def __init__(self, index_path: str = '/home/dfoadmin/boqu/insightful_summarization/simple_context_index.pkl'):
        self.index_path = index_path
        self.corpora: Dict[str, List[str]] = {'scientific_paper': [], 'news': [], 'article': []}
        self.tfidf: Dict[str, Dict[str, Dict[str, float]]] = {'scientific_paper': {}, 'news': {}, 'article': {}}
        self.load_index()

    def load_index(self) -> None:
        """Load the index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    self.corpora, self.tfidf = pickle.load(f)
                logger.info(f"Index loaded from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")

    def save_index(self) -> None:
        """Save the index to disk."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump((self.corpora, self.tfidf), f)
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def add_documents(self, documents: List[str], domain: str) -> None:
        """Add new documents to the index for a specific domain."""
        self.corpora[domain].extend(documents)
        self._update_tfidf(domain)
        logger.info(f"Added {len(documents)} documents to {domain} domain")

    def _update_tfidf(self, domain: str) -> None:
        """Update TF-IDF scores for a specific domain."""
        documents = self.corpora[domain]
        word_doc_count = Counter()
        for doc in documents:
            word_doc_count.update(set(doc.lower().split()))

        for i, doc in enumerate(documents):
            word_counts = Counter(doc.lower().split())
            for word, count in word_counts.items():
                tf = count / len(doc.split())
                idf = math.log(len(documents) / (word_doc_count[word] + 1))
                self.tfidf[domain].setdefault(word, {})[i] = tf * idf

    def retrieve(self, query: str, domain: str, k: int = 5) -> List[str]:
        """Retrieve the k most relevant documents for the query from a specific domain."""
        query_words = set(query.lower().split())
        scores = [0] * len(self.corpora[domain])
        for word in query_words:
            if word in self.tfidf[domain]:
                for doc_id, score in self.tfidf[domain][word].items():
                    scores[doc_id] += score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.corpora[domain][i] for i in top_indices]

def retrieve_relevant_context(document: str, domain: str, k: int = 5) -> List[str]:
    """
    Retrieve relevant context for the given document based on its domain.
    
    Args:
    document (str): The text of the document
    domain (str): The domain of the document ('scientific_paper', 'news', or 'article')
    k (int): Number of relevant contexts to retrieve
    
    Returns:
    List[str]: A list of relevant context snippets
    """
    retriever = SimpleContextRetriever()

    # If the index for this domain is empty, add some sample documents
    if len(retriever.corpora[domain]) == 0:
        sample_corpora = {
            'scientific_paper': [
                "The discovery of CRISPR-Cas9 revolutionized gene editing techniques.",
                "Quantum entanglement was first proposed by Einstein, Podolsky, and Rosen in 1935.",
                "The Human Genome Project completed the sequencing of human DNA in 2003.",
                "The detection of gravitational waves confirmed a major prediction of Einstein's general theory of relativity.",
                "The development of mRNA vaccines marked a breakthrough in immunology and virology."
            ],
            'news': [
                "The fall of the Berlin Wall in 1989 marked the end of the Cold War era.",
                "The 9/11 attacks in 2001 reshaped global geopolitics and security policies.",
                "The 2008 financial crisis led to widespread economic reforms and regulations.",
                "The COVID-19 pandemic in 2020 caused global lockdowns and economic disruption.",
                "The Paris Agreement of 2015 set international targets for reducing greenhouse gas emissions."
            ],
            'article': [
                "The rise of social media has transformed communication and information spread.",
                "Artificial intelligence poses both opportunities and challenges for the job market.",
                "The growing wealth gap is a pressing issue in many developed countries.",
                "Climate change denial hinders effective environmental policy implementation.",
                "The ethics of gene editing in humans is a contentious topic in bioethics."
            ]
        }
        retriever.add_documents(sample_corpora[domain], domain)
        retriever.save_index()

    # Retrieve relevant context
    relevant_contexts = retriever.retrieve(document, domain, k)
    return relevant_contexts

def main():
    documents = {
        'scientific_paper': """
        Recent advancements in CRISPR technology have allowed for more precise gene editing in human embryos.
        This breakthrough builds upon the initial discovery of CRISPR-Cas9 and addresses previous concerns about off-target effects.
        """,
        'news': """
        A major earthquake struck the coast of Japan today, triggering tsunami warnings across the Pacific.
        This event echoes the devastating 2011 Tohoku earthquake and tsunami, raising concerns about nuclear plant safety.
        """,
        'article': """
        The rapid advancement of artificial intelligence in recent years has sparked debates about its impact on employment.
        While AI promises increased efficiency, there are growing concerns about job displacement across various sectors.
        """
    }

    for domain, doc in documents.items():
        logger.info(f"\nRetrieving context for {domain}:")
        contexts = retrieve_relevant_context(doc, domain)
        for i, context in enumerate(contexts, 1):
            logger.info(f"{i}. {context}")

if __name__ == "__main__":
    main()