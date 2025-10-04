from transformers import pipeline
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class NLPProcessor:
    def __init__(self, config):
        self.config = config
        print("Loading AI models...")
        self.summarizer = pipeline("summarization", model=config.SUMMARIZER_MODEL)
        self.embedder = SentenceTransformer(config.EMBEDDINGS_MODEL)
        try:
            self.nlp = spacy.load(config.NER_MODEL)
        except:
            print("Scientific NER not found, using default")
            self.nlp = spacy.load("en_core_web_sm")
    
    def generate_summary(self, text, max_length=150):
        """Generate AI summary of paper"""
        if not text or len(text) < 50:
            return "Summary not available"
        
        # Check word count
        word_count = len(text.split())
        
        # If text is too short, return as-is
        if word_count < 50:
            return text
        
        try:
            # Adaptive max_length based on input
            max_length = min(max_length, max(30, word_count // 2))
            min_length = max(10, max_length // 3)
            
            # Truncate to model limits
            text = text[:1024]
            summary = self.summarizer(text, 
                                     max_length=max_length, 
                                     min_length=min_length, 
                                     do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback: return first 200 chars
            return text[:200] + "..."
    
    def extract_entities(self, text):
        """Extract biological entities using NER"""
        doc = self.nlp(text[:100000])  # Spacy has token limits
        
        entities = {
            'organisms': [],
            'body_systems': [],
            'biological_processes': [],
            'chemicals': [],
            'environments': []
        }
        
        # Define entity categories
        organism_keywords = ['plant', 'mouse', 'rat', 'human', 'cell', 'bacteria', 'yeast']
        environment_keywords = ['microgravity', 'space', 'ISS', 'radiation', 'Mars', 'Moon']
        
        for ent in doc.ents:
            text_lower = ent.text.lower()
            
            if ent.label_ in ['ORG', 'PERSON']:
                continue
            
            if any(kw in text_lower for kw in organism_keywords):
                entities['organisms'].append(ent.text)
            elif any(kw in text_lower for kw in environment_keywords):
                entities['environments'].append(ent.text)
            elif ent.label_ == 'CHEMICAL':
                entities['chemicals'].append(ent.text)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # Top 10
        
        return entities
    
    def generate_embeddings(self, texts):
        """Generate vector embeddings for semantic search"""
        return self.embedder.encode(texts, show_progress_bar=True)
    
    def identify_topics(self, papers, n_topics=10):
        """Cluster papers into topics using embeddings"""
        texts = [p.get('abstract', '') or p.get('full_text', '')[:500] for p in papers]
        embeddings = self.generate_embeddings(texts)
        
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Extract keywords for each cluster
        topics = []
        for i in range(n_topics):
            cluster_papers = [papers[j] for j in range(len(papers)) if clusters[j] == i]
            cluster_titles = [p['title'] for p in cluster_papers]
            
            # Simple keyword extraction from titles
            words = ' '.join(cluster_titles).lower().split()
            common_words = Counter(words).most_common(5)
            topic_label = ' '.join([w[0] for w in common_words])
            
            topics.append({
                'id': i,
                'label': topic_label,
                'papers': cluster_papers,
                'count': len(cluster_papers)
            })
        
        return topics