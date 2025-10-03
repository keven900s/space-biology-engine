import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, nlp_processor):
        self.nlp = nlp_processor
        self.paper_embeddings = None
        self.papers = None
    
    def index_papers(self, papers):
        """Create vector index for all papers"""
        self.papers = papers
        texts = [self._get_searchable_text(p) for p in papers]
        self.paper_embeddings = self.nlp.generate_embeddings(texts)
        print(f"Indexed {len(papers)} papers for semantic search")
    
    def _get_searchable_text(self, paper):
        """Combine relevant fields for search"""
        parts = [
            paper.get('title', ''),
            paper.get('abstract', ''),
            paper.get('conclusion', '')
        ]
        return ' '.join([p for p in parts if p])[:500]
    
    def search(self, query, top_k=10):
        """Semantic search using query embedding"""
        query_embedding = self.nlp.embedder.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.paper_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'paper': self.papers[idx],
                'similarity': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
