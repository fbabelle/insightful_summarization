import os
import sys
import logging
from datetime import datetime
import pickle
from typing import List, Dict, Tuple, Union, Optional, Any
import hashlib
import json
import uuid
import math
from collections import Counter
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from __init__ import root_dir

# Set up log directory and file
log_dir = os.path.join(root_dir, 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'retrieved_history.log')

print(f"Log directory: {log_dir}")
print(f"Log file: {log_file}")

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        print(f"File handler added for {log_file}")
    except Exception as e:
        print(f"Error setting up file handler: {e}")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Set up logging
logger = setup_logging()



@dataclass
class EmbeddingConfig:
    model_name: str = 'avsolatorio/NoInstruct-small-Embedding-v0'
    batch_size: int = 32
    max_length: int = 512
    fallback_to_single: bool = True
    min_batch_size: int = 4
    master_port: str = '12355'

class DistributedEmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.is_distributed = self.num_gpus > 1
        
        # Initialize model and tokenizer for single-device fallback
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Use half precision for GPU
        
        self.logger.info(f"Initialized with {self.num_gpus} GPUs. Distributed: {self.is_distributed}")

    def _split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into balanced chunks for each GPU."""
        if not texts:
            return []
        
        n = len(texts)
        chunk_size = math.ceil(n / self.num_gpus)
        return [texts[i:i + chunk_size] for i in range(0, n, chunk_size)]

    def _process_batch(self, texts: List[str], device: torch.device) -> np.ndarray:
        """Process a single batch of texts on the specified device."""
        if not texts:
            return np.array([])

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

    def _process_chunks(self, texts: List[str], rank: int) -> np.ndarray:
        """Process chunks of texts on a single GPU."""
        if not texts:
            return np.array([])

        device = torch.device(f'cuda:{rank}')
        embeddings_list = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            if batch:  # Check if batch is not empty
                batch_embeddings = self._process_batch(batch, device)
                if batch_embeddings.size > 0:  # Check if embeddings were generated
                    embeddings_list.append(batch_embeddings)

        return np.vstack(embeddings_list) if embeddings_list else np.array([])

    def _single_device_generation(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using a single device."""
        if not texts:
            return np.array([])

        embeddings_list = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            if batch:
                batch_embeddings = self._process_batch(batch, self.device)
                if batch_embeddings.size > 0:
                    embeddings_list.append(batch_embeddings)

        return np.vstack(embeddings_list) if embeddings_list else np.array([])

    def _distributed_generation(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using multiple GPUs."""
        if len(texts) < self.config.min_batch_size * self.num_gpus:
            self.logger.info("Small batch detected, falling back to single device")
            return self._single_device_generation(texts)

        try:
            # Split texts into chunks for each GPU
            text_chunks = self._split_texts(texts)
            
            # Process each chunk on a different GPU
            embeddings_list = []
            for rank, chunk in enumerate(text_chunks):
                if chunk:  # Only process non-empty chunks
                    embeddings = self._process_chunks(chunk, rank)
                    if embeddings.size > 0:
                        embeddings_list.append(embeddings)

            return np.vstack(embeddings_list) if embeddings_list else np.array([])

        except Exception as e:
            self.logger.error(f"Error in distributed generation: {str(e)}")
            if self.config.fallback_to_single:
                self.logger.info("Falling back to single device generation")
                return self._single_device_generation(texts)
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the most appropriate method."""
        if not texts:
            return np.array([])

        try:
            if self.is_distributed and len(texts) >= self.config.min_batch_size * self.num_gpus:
                return self._distributed_generation(texts)
            return self._single_device_generation(texts)
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            if self.config.fallback_to_single:
                self.logger.info("Falling back to single device generation")
                return self._single_device_generation(texts)
            raise



class EmbeddingGenerator:
    def __init__(self, model_name='avsolatorio/NoInstruct-small-Embedding-v0'):
        config = EmbeddingConfig(
            model_name=model_name,
            batch_size=32,
            max_length=512,
            fallback_to_single=True,
            min_batch_size=4
        )
        self.generator = DistributedEmbeddingGenerator(config)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings with automatic batch size adjustment."""
        if not texts:
            return np.array([])
            
        try:
            return self.generator.generate_embeddings(texts)
        except Exception as e:
            self.logger.error(f"Error in embedding generation: {str(e)}")
            return np.array([])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cluster_with_threshold(embeddings: np.ndarray, piece_ids: List[str], threshold: float) -> List[List[int]]:
    clusters = []
    id_to_cluster = {}
    for i, (emb, piece_id) in enumerate(zip(embeddings, piece_ids)):
        base_id = piece_id.split('_')[0]  # Assuming piece_id format: "base_id_piece_number"
        if base_id in id_to_cluster:
            clusters[id_to_cluster[base_id]].append(i)
        else:
            assigned = False
            for j, cluster in enumerate(clusters):
                if cosine_similarity(emb, embeddings[cluster[0]]) > threshold:
                    cluster.append(i)
                    id_to_cluster[base_id] = j
                    assigned = True
                    break
            if not assigned:
                id_to_cluster[base_id] = len(clusters)
                clusters.append([i])
    return clusters

def improved_clustering(embeddings: np.ndarray, min_clusters: int = 5, max_clusters: int = None, initial_threshold: float = 0.9) -> List[List[int]]:
    if max_clusters is None:
        max_clusters = len(embeddings) // 10

    # Initial clustering
    clusters = cluster_with_threshold(embeddings, initial_threshold)

    # If we have more than max_clusters, merge until we reach max_clusters
    while len(clusters) > max_clusters:
        # Find the two most similar clusters
        max_similarity = -1
        merge_indices = (-1, -1)
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters[i+1:], i+1):
                sim = cosine_similarity(embeddings[c1[0]], embeddings[c2[0]])
                if sim > max_similarity:
                    max_similarity = sim
                    merge_indices = (i, j)
        
        # Merge the most similar clusters
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    # If we have fewer than min_clusters, try to split by lowering the threshold
    if len(clusters) < min_clusters:
        thresholds_to_try = [initial_threshold + (1 - initial_threshold) * i / 4 for i in range(1, 4)]
        
        for new_threshold in thresholds_to_try:
            new_clusters = cluster_with_threshold(embeddings, new_threshold)
            if len(new_clusters) >= min_clusters:
                return new_clusters
            elif len(new_clusters) > len(clusters):
                clusters = new_clusters  # Keep the improvement, but continue trying

    return clusters

class EnhancedContextRetriever:
    def __init__(self, index_path: str = os.path.join(root_dir, 'embedding/content_embeddings.pkl')):
        logger.info("Initializing EnhancedContextRetriever")
        self.index_path = index_path
        self.id_usage_path = os.path.join(os.path.dirname(index_path), 'id_usage.json')
        self.corpora: Dict[str, List[Tuple[str, str, str, str]]] = {}  # (title, content, piece_id, unique_id)
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.clusters: Dict[str, List[List[int]]] = {}
        self.id_usage: Dict[str, str] = {}  # hash -> unique_id
        self.embedding_generator = EmbeddingGenerator()
        self.load_index()
        self.load_id_usage()

    def load_index(self) -> None:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    self.corpora, self.embeddings, self.clusters = pickle.load(f)
                logger.info(f"Index loaded from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.corpora, self.embeddings, self.clusters = {}, {}, {}

    def save_index(self) -> None:
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump((self.corpora, self.embeddings, self.clusters), f)
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def load_id_usage(self) -> None:
        if os.path.exists(self.id_usage_path):
            try:
                with open(self.id_usage_path, 'r') as f:
                    self.id_usage = json.load(f)
                logger.info(f"ID usage loaded from {self.id_usage_path}")
            except Exception as e:
                logger.error(f"Error loading ID usage: {str(e)}")
                self.id_usage = {}
        else:
            self.id_usage = {}
            self.rebuild_id_usage()

    def save_id_usage(self) -> None:
        try:
            with open(self.id_usage_path, 'w') as f:
                json.dump(self.id_usage, f)
            logger.info(f"ID usage saved to {self.id_usage_path}")
        except Exception as e:
            logger.error(f"Error saving ID usage: {str(e)}")

    def rebuild_id_usage(self) -> None:
        self.id_usage = {}
        for domain, docs in self.corpora.items():
            for title, content, piece_id, unique_id in docs:
                doc_hash = self.generate_hash(title, content)
                self.id_usage[doc_hash] = unique_id
        self.save_id_usage()

    def generate_hash(self, title: str, content: str) -> str:
        return hashlib.md5((title + content).encode()).hexdigest()

    def generate_unique_id(self) -> str:
        return uuid.uuid4().hex

    def ensure_domain_initialized(self, domain: str) -> None:
        if domain not in self.corpora:
            self.corpora[domain] = []
        if domain not in self.embeddings:
            self.embeddings[domain] = {'title': None, 'content': None}
        if domain not in self.clusters:
            self.clusters[domain] = []

    def add_documents(self, documents: List[Tuple[str, str]], domain: str, max_piece_length: int = 512) -> Dict[str, List[Tuple[str, str, str]]]:
        self.ensure_domain_initialized(domain)

        new_titles = []
        new_contents = []
        new_piece_ids = []
        new_unique_ids = []
        added_docs = []
        skipped_docs = []

        for doc_id, (title, content) in enumerate(documents):
            pieces = [content[i:i+max_piece_length] for i in range(0, len(content), max_piece_length)]
            doc_added = False
            for piece_num, piece in enumerate(pieces):
                piece_id = f"{doc_id}_{piece_num}"
                doc_hash = self.generate_hash(title, piece)
                
                if doc_hash in self.id_usage:
                    # Document already exists
                    unique_id = self.id_usage[doc_hash]
                    skipped_docs.append((title, piece, unique_id))
                    logger.info(f"Skipped duplicate document: {title} (piece {piece_num + 1}/{len(pieces)})")
                else:
                    # New document
                    unique_id = self.generate_unique_id()
                    self.id_usage[doc_hash] = unique_id
                    self.corpora[domain].append((title, piece, piece_id, unique_id))
                    new_titles.append(title)
                    new_contents.append(piece)
                    new_piece_ids.append(piece_id)
                    new_unique_ids.append(unique_id)
                    added_docs.append((title, piece, unique_id))
                    doc_added = True

            if doc_added:
                logger.info(f"Added document: {title} ({len(pieces)} pieces)")

        if new_contents:
            title_embeddings = self.embedding_generator.generate_embeddings(new_titles)
            content_embeddings = self.embedding_generator.generate_embeddings(new_contents)

            if self.embeddings[domain]['title'] is None:
                self.embeddings[domain]['title'] = title_embeddings
                self.embeddings[domain]['content'] = content_embeddings
            else:
                self.embeddings[domain]['title'] = np.vstack([self.embeddings[domain]['title'], title_embeddings])
                self.embeddings[domain]['content'] = np.vstack([self.embeddings[domain]['content'], content_embeddings])

            self._update_clusters(domain)

        self.save_id_usage()
        logger.info(f"Added {len(added_docs)} new documents ({len(new_piece_ids)} pieces) to {domain} domain")
        logger.info(f"Skipped {len(skipped_docs)} duplicate documents")

        return {
            "added": added_docs,
            "skipped": skipped_docs
        }

    def get_documents(self, domain: str, identifier: Union[str, int]) -> List[Tuple[str, str, str, str]]:
        """
        Retrieve documents from a specific domain based on title or index.
        
        :param domain: The domain to search in
        :param identifier: Either the title (str) or index (int) of the document(s) to retrieve
        :return: A list of tuples (title, content, piece_id, unique_id) matching the identifier
        """
        if domain not in self.corpora:
            logger.warning(f"Domain '{domain}' not found.")
            return []

        results = []
        if isinstance(identifier, str):
            # Search by title
            results = [doc for doc in self.corpora[domain] if doc[0] == identifier]
        elif isinstance(identifier, int):
            # Search by index
            if 0 <= identifier < len(self.corpora[domain]):
                results = [self.corpora[domain][identifier]]
            else:
                logger.warning(f"Index {identifier} is out of range for domain '{domain}'.")
        else:
            logger.warning(f"Invalid identifier type: {type(identifier)}. Must be str or int.")

        logger.info(f"Retrieved {len(results)} documents from domain '{domain}' using identifier: {identifier}")
        return results

    def remove_documents(self, domain: str, identifiers: Union[List[str], List[int]]) -> None:
        if domain not in self.corpora:
            logger.warning(f"Domain '{domain}' not found. No documents removed.")
            return

        indices_to_remove = set()
        for identifier in identifiers:
            if isinstance(identifier, str):
                # If identifier is a title (string)
                indices = [i for i, (title, _, _, _) in enumerate(self.corpora[domain]) if title == identifier]
                indices_to_remove.update(indices)
            elif isinstance(identifier, int):
                # If identifier is an index (integer)
                if 0 <= identifier < len(self.corpora[domain]):
                    indices_to_remove.add(identifier)
                else:
                    logger.warning(f"Index {identifier} is out of range for domain '{domain}'. Skipping.")
            else:
                logger.warning(f"Invalid identifier type: {type(identifier)}. Skipping.")

        if not indices_to_remove:
            logger.warning(f"No valid documents found to remove from domain '{domain}'.")
            return

        # Remove documents from corpora
        self.corpora[domain] = [doc for i, doc in enumerate(self.corpora[domain]) if i not in indices_to_remove]

        # Remove corresponding embeddings
        mask = np.ones(len(self.embeddings[domain]['title']), dtype=bool)
        mask[list(indices_to_remove)] = False
        self.embeddings[domain]['title'] = self.embeddings[domain]['title'][mask]
        self.embeddings[domain]['content'] = self.embeddings[domain]['content'][mask]

        # Update clusters
        self._update_clusters(domain)

        logger.info(f"Removed {len(indices_to_remove)} documents from domain '{domain}'.")

    def remove_domain(self, domain: str) -> None:
        if domain in self.corpora:
            del self.corpora[domain]
            del self.embeddings[domain]
            del self.clusters[domain]
            logger.info(f"Domain '{domain}' and all its associated data have been removed.")
        else:
            logger.warning(f"Domain '{domain}' not found. No data removed.")

    def _update_clusters(self, domain: str) -> None:
        if self.embeddings[domain]['content'].size > 0:
            self.clusters[domain] = cluster_with_threshold(self.embeddings[domain]['content'], 
                                                           [doc[2] for doc in self.corpora[domain]], 
                                                           threshold=0.9)
        else:
            self.clusters[domain] = []

    def retrieve(self, input_title: str, input_content: str, domain: str, k: int = 5, title_weight: float = 0.5, alpha: float = 0.5, cluster_penalty: float = 0.2, content_penalty: float = 0.4) -> List[Tuple[str, str, str]]:
        self.ensure_domain_initialized(domain)
        
        if len(self.corpora[domain]) == 0:
            logger.warning(f"No documents found for domain: {domain}")
            return []

        # Generate embeddings for input title and content
        input_title_embedding = self.embedding_generator.generate_embeddings([input_title])[0]
        input_content_embedding = self.embedding_generator.generate_embeddings([input_content])[0]

        # Calculate similarities for titles and contents
        title_similarities = np.array([cosine_similarity(input_title_embedding, doc_emb) for doc_emb in self.embeddings[domain]['title']])
        content_similarities = np.array([cosine_similarity(input_content_embedding, doc_emb) for doc_emb in self.embeddings[domain]['content']])
        
        # Combine title and content similarities using title_weight
        combined_similarities = title_weight * title_similarities + (1 - title_weight) * content_similarities
        
        max_matches = min(k * 4, len(self.corpora[domain]))
        top_matches = np.argsort(combined_similarities)[-max_matches:][::-1]
        
        cluster_counter = Counter()
        content_counter = Counter()
        final_matches = []
        
        for i in top_matches:
            cluster = next((idx for idx, cluster in enumerate(self.clusters[domain]) if i in cluster), -1)
            base_id = self.corpora[domain][i][2].split('_')[0]
            
            score = combined_similarities[i]
            if cluster != -1 and cluster_counter[cluster] > k // 4:
                score *= (1 - cluster_penalty)
            
            if content_counter[base_id] >= 1:
                score *= (1 - content_penalty)
            
            final_matches.append((i, score, base_id))
            cluster_counter[cluster] += 1
            content_counter[base_id] += 1
            
            if len(set(match[2] for match in final_matches)) == k:
                break
        
        final_matches.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        seen_base_ids = set()
        for i, _, base_id in final_matches:
            if base_id not in seen_base_ids:
                title, content, _, unique_id = self.corpora[domain][i]
                full_content = " ".join(piece for _, piece, piece_id, _ in self.corpora[domain] if piece_id.startswith(base_id))
                results.append((title, full_content, unique_id))
                seen_base_ids.add(base_id)
            
            if len(results) == k:
                break
        
        logger.info(f"Retrieved {len(results)} documents for input in domain '{domain}'")
        return results

    def get_summary(self) -> str:
        summary = []
        total_chunks = sum(len(docs) for docs in self.corpora.values())
        
        summary.append(f"Total number of domains: {len(self.corpora)}")
        summary.append(f"Total number of chunks across all domains: {total_chunks}")
        
        for domain, docs in self.corpora.items():
            domain_chunks = len(docs)
            domain_percentage = (domain_chunks / total_chunks) * 100 if total_chunks > 0 else 0
            summary.append(f"\nDomain: {domain}")
            summary.append(f"  Number of chunks: {domain_chunks} ({domain_percentage:.2f}% of total)")
            
            if domain in self.clusters:
                num_clusters = len(self.clusters[domain])
                summary.append(f"  Number of clusters: {num_clusters}")
                
                for i, cluster in enumerate(self.clusters[domain], 1):
                    cluster_size = len(cluster)
                    cluster_percentage = (cluster_size / domain_chunks) * 100
                    summary.append(f"    Cluster {i}: {cluster_size} chunks ({cluster_percentage:.2f}% of domain)")
            else:
                summary.append("  No clusters found for this domain")
        
        logger.info("Generated summary of EnhancedContextRetriever")
        return "\n".join(summary)

def retrieve_relevant_context(title: str, content: str, domain: str, k: int = 5, title_weight: float = 0.5) -> List[Tuple[str, str, str]]:
    retriever = EnhancedContextRetriever()
    retriever.ensure_domain_initialized(domain)

    if len(retriever.corpora[domain]) == 0:
        logger.info(f"No documents found for domain: {domain}. Adding sample documents.")
        sample_corpora = {
            'scientific_research_paper': [
                ("CRISPR-Cas9 Discovery", "The discovery of CRISPR-Cas9 revolutionized gene editing techniques."),
                ("Quantum Entanglement", "Quantum entanglement was first proposed by Einstein, Podolsky, and Rosen in 1935."),
                ("Human Genome Project", "The Human Genome Project completed the sequencing of human DNA in 2003."),
            ],
            'news': [
                ("Berlin Wall Fall", "The fall of the Berlin Wall in 1989 marked the end of the Cold War era."),
                ("9/11 Attacks", "The 9/11 attacks in 2001 reshaped global geopolitics and security policies."),
                ("COVID-19 Pandemic", "The COVID-19 pandemic in 2020 caused global lockdowns and economic disruption."),
            ],
            'article': [
                ("Social Media Impact", "The rise of social media has transformed communication and information spread."),
                ("AI and Employment", "Artificial intelligence poses both opportunities and challenges for the job market."),
                ("Climate Change", "Climate change is a pressing global issue requiring immediate action and policy changes."),
            ]
        }
        
        if domain in sample_corpora:
            documents_to_add = sample_corpora[domain]
        else:
            # Create placeholder content for new domains
            logger.warning(f"Creating placeholder content for new domain: {domain}")
            documents_to_add = [
                (f"{domain.capitalize()} Topic 1", f"This is a placeholder document for the {domain} domain."),
                (f"{domain.capitalize()} Topic 2", f"Another placeholder document for the {domain} domain."),
                (f"{domain.capitalize()} Topic 3", f"A third placeholder document for the {domain} domain."),
            ]
        retriever.add_documents(documents_to_add, domain)

    relevant_contexts = retriever.retrieve(title, content, domain, k=k, title_weight=title_weight)

    # Add the input document to the retriever
    new_docs = [(title, content)]
    retriever.add_documents(new_docs, domain)
    
    retriever.save_index()
    return relevant_contexts


def main():
    logger.info("Starting main function")
    retriever = EnhancedContextRetriever()
    
    # Add some sample documents
    sample_docs = {
        'scientific_paper': [
            ("CRISPR-Cas9 Discovery", "The discovery of CRISPR-Cas9 revolutionized gene editing techniques."),
            ("Quantum Entanglement", "Quantum entanglement was first proposed by Einstein, Podolsky, and Rosen in 1935."),
            ("Human Genome Project", "The Human Genome Project completed the sequencing of human DNA in 2003."),
        ],
        'news': [
            ("Berlin Wall Fall", "The fall of the Berlin Wall in 1989 marked the end of the Cold War era."),
            ("9/11 Attacks", "The 9/11 attacks in 2001 reshaped global geopolitics and security policies."),
        ],
        'article': [
            ("Social Media Impact", "The rise of social media has transformed communication and information spread."),
            ("AI and Employment", "Artificial intelligence poses both opportunities and challenges for the job market."),
            ("Climate Change", "Climate change is a pressing global issue requiring immediate action and policy changes."),
        ]
    }
    
    for domain, docs in sample_docs.items():
        logger.info(f"Adding documents to domain: {domain}")
        retriever.add_documents(docs, domain)
    
    # Perform some retrievals
    documents = {
        'scientific_paper':
            ("title1","Recent advancements in CRISPR technology have allowed for more precise gene editing in human embryos."),
        'news':
            ("title2","A major earthquake struck the coast of Japan today, triggering tsunami warnings across the Pacific."),
        'article':
            ("title3","The rapid advancement of artificial intelligence in recent years has sparked debates about its impact on employment."),
    }

    for domain, doc in documents.items():
        logger.info(f"\nRetrieving context for {domain}:")
        title, content = doc
        contexts = retriever.retrieve(title, content, domain, k=2)
        for i, (title, content, unique_id) in enumerate(contexts, 1):
            logger.info(f"{i}. Title: {title}")
            logger.info(f"   Unique ID: {unique_id}")
            logger.info(f"   Content: {content[:100]}...")  # Truncated for brevity
            logger.info("---")

    # Print summary
    logger.info("\nSummary of EnhancedContextRetriever:")
    logger.info(retriever.get_summary())

    # Test get_documents function
    logger.info("\nTesting get_documents function:")
    docs = retriever.get_documents('scientific_paper', "CRISPR-Cas9 Discovery")
    for doc in docs:
        logger.info(f"Title: {doc[0]}")
        logger.info(f"Content: {doc[1][:100]}...")  # Truncated for brevity
        logger.info(f"Piece ID: {doc[2]}")
        logger.info(f"Unique ID: {doc[3]}")
        logger.info("---")

    # test remove_documents function
    logger.info("\nTesting remove_documents function:")
    retriever.remove_documents('news', ["Berlin Wall Fall", 1])
    
    # test remove_domain function
    logger.info("\nTesting remove_domain function:")
    retriever.remove_domain('new_domain')

    # get summary again
    logger.info("\nSummary of EnhancedContextRetriever after removal:")
    logger.info(retriever.get_summary())

    logger.info("Main function completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred in the main function")
    finally:
        # Ensure all logs are written before the script exits
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logging.shutdown()

print("Script execution completed. Please check the log file for details.")