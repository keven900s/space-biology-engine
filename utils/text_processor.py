# utils/text_processor.py
# Text processing utilities for NASA Space Biology Knowledge Engine

import re
import string
from typing import List, Dict, Set, Tuple
from collections import Counter
import numpy as np

class TextProcessor:
    """
    Comprehensive text processing utilities for cleaning, normalizing,
    and extracting information from scientific papers
    """
    
    def __init__(self):
        # Scientific stopwords (common words to filter out)
        self.stopwords = self._load_stopwords()
        
        # Domain-specific keywords for space biology
        self.space_biology_terms = self._load_space_biology_terms()
        
        # Abbreviations dictionary
        self.abbreviations = self._load_abbreviations()
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords including scientific terms"""
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        scientific_stopwords = {
            'study', 'research', 'paper', 'article', 'results', 'conclusion',
            'abstract', 'introduction', 'methods', 'discussion', 'analysis',
            'data', 'figure', 'table', 'showed', 'found', 'observed', 'reported',
            'demonstrated', 'indicated', 'suggested', 'measured', 'examined',
            'investigated', 'analyzed', 'evaluated', 'assessed', 'determined',
            'used', 'using', 'based', 'shown', 'present', 'presents'
        }
        
        return basic_stopwords | scientific_stopwords
    
    def _load_space_biology_terms(self) -> Dict[str, List[str]]:
        """Load domain-specific terms for entity extraction"""
        return {
            'organisms': [
                'human', 'humans', 'mouse', 'mice', 'rat', 'rats',
                'plant', 'plants', 'arabidopsis', 'bacteria', 'yeast',
                'drosophila', 'c. elegans', 'cell', 'cells', 'tissue',
                'escherichia coli', 'saccharomyces cerevisiae'
            ],
            'environments': [
                'microgravity', 'space', 'spaceflight', 'iss', 'international space station',
                'radiation', 'cosmic radiation', 'solar radiation', 'mars', 'lunar', 'moon',
                'parabolic flight', 'ground control', 'analog', 'simulated microgravity',
                'clinostat', 'rotating wall vessel', 'hindlimb unloading'
            ],
            'body_systems': [
                'cardiovascular', 'musculoskeletal', 'nervous', 'immune', 'skeletal',
                'respiratory', 'digestive', 'endocrine', 'reproductive', 'integumentary',
                'bone', 'muscle', 'heart', 'brain', 'lung', 'kidney', 'liver'
            ],
            'biological_processes': [
                'gene expression', 'cell division', 'apoptosis', 'metabolism',
                'differentiation', 'proliferation', 'growth', 'development',
                'adaptation', 'signaling', 'transcription', 'translation',
                'protein synthesis', 'dna repair', 'cell cycle', 'mitosis'
            ],
            'effects': [
                'atrophy', 'hypertrophy', 'loss', 'gain', 'increase', 'decrease',
                'damage', 'repair', 'adaptation', 'dysfunction', 'impairment',
                'enhancement', 'reduction', 'elevation', 'suppression'
            ],
            'countermeasures': [
                'exercise', 'resistance training', 'nutrition', 'pharmacological',
                'artificial gravity', 'centrifuge', 'vibration', 'supplement'
            ],
            'missions': [
                'apollo', 'shuttle', 'skylab', 'mir', 'iss', 'spacex',
                'artemis', 'mars mission', 'lunar mission', 'dragon', 'soyuz'
            ]
        }
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load common scientific abbreviations"""
        return {
            'ISS': 'International Space Station',
            'NASA': 'National Aeronautics and Space Administration',
            'ESA': 'European Space Agency',
            'JAXA': 'Japan Aerospace Exploration Agency',
            'DNA': 'Deoxyribonucleic Acid',
            'RNA': 'Ribonucleic Acid',
            'PCR': 'Polymerase Chain Reaction',
            'MRI': 'Magnetic Resonance Imaging',
            'CT': 'Computed Tomography',
            'BMD': 'Bone Mineral Density',
            'VO2': 'Oxygen Consumption',
            'LEO': 'Low Earth Orbit',
            'EVA': 'Extravehicular Activity',
            'ECLSS': 'Environmental Control and Life Support System'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep periods for sentences
        text = re.sub(r'[^a-zA-Z0-9\s\.\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words
        """
        text = self.clean_text(text)
        
        # Split into words
        tokens = text.split()
        
        # Filter out very short words
        tokens = [t for t in tokens if len(t) > 2]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s) > 20]  # Filter very short sentences
        
        return sentences
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract top keywords from text using frequency analysis
        """
        tokens = self.tokenize(text, remove_stopwords=True)
        
        # Count frequencies
        freq = Counter(tokens)
        
        # Get top N
        top_keywords = freq.most_common(top_n)
        
        return top_keywords
    
    def extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """
        Extract common phrases (n-grams) from text
        """
        tokens = self.tokenize(text, remove_stopwords=False)
        phrases = []
        
        for n in range(min_length, max_length + 1):
            for i in range(len(tokens) - n + 1):
                phrase = ' '.join(tokens[i:i+n])
                # Keep phrases with at least one non-stopword
                if any(t not in self.stopwords for t in tokens[i:i+n]):
                    phrases.append(phrase)
        
        # Count and return most common
        phrase_freq = Counter(phrases)
        return [phrase for phrase, _ in phrase_freq.most_common(20)]
    
    def extract_domain_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract domain-specific entities from text
        """
        text_lower = text.lower()
        found_entities = {}
        
        for entity_type, terms in self.space_biology_terms.items():
            found = []
            for term in terms:
                if term in text_lower:
                    found.append(term)
            
            if found:
                found_entities[entity_type] = list(set(found))  # Remove duplicates
        
        return found_entities
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand known abbreviations in text
        """
        for abbr, full in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, f"{abbr} ({full})", text, count=1)
        
        return text
    
    def extract_numeric_data(self, text: str) -> List[Dict[str, any]]:
        """
        Extract numeric data and their contexts from text
        """
        # Pattern for numbers with optional units
        pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?'
        
        matches = re.finditer(pattern, text)
        numeric_data = []
        
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else ''
            
            # Get surrounding context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            
            numeric_data.append({
                'value': value,
                'unit': unit,
                'context': context
            })
        
        return numeric_data
    
    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics (Flesch Reading Ease)
        """
        sentences = self.extract_sentences(text)
        words = self.tokenize(text, remove_stopwords=False)
        
        if not sentences or not words:
            return {'flesch_score': 0, 'reading_level': 'Unknown'}
        
        # Count syllables (simplified)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiou'
            count = 0
            prev_char_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = is_vowel
            
            return max(1, count)  # At least 1 syllable per word
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        # Interpret score
        if flesch_score >= 90:
            level = 'Very Easy'
        elif flesch_score >= 80:
            level = 'Easy'
        elif flesch_score >= 70:
            level = 'Fairly Easy'
        elif flesch_score >= 60:
            level = 'Standard'
        elif flesch_score >= 50:
            level = 'Fairly Difficult'
        elif flesch_score >= 30:
            level = 'Difficult'
        else:
            level = 'Very Difficult'
        
        return {
            'flesch_score': round(flesch_score, 2),
            'reading_level': level,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2)
        }
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract citation patterns from text
        """
        # Common citation patterns
        patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.\s+\d{4}\)',  # (Smith et al. 2020)
            r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+\d{4}\)',  # (Smith and Jones 2020)
            r'\[[0-9,\-\s]+\]',  # [1, 2, 3-5]
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compare two texts for similarity
        """
        # Tokenize both texts
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        jaccard = len(intersection) / len(union) if union else 0
        
        # Overlap coefficient
        overlap = len(intersection) / min(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0
        
        # Common keywords
        common_keywords = list(intersection)[:10]
        
        return {
            'jaccard_similarity': round(jaccard, 3),
            'overlap_coefficient': round(overlap, 3),
            'common_keywords': common_keywords
        }
    
    def get_text_statistics(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive text statistics
        """
        sentences = self.extract_sentences(text)
        words = self.tokenize(text, remove_stopwords=False)
        words_no_stop = self.tokenize(text, remove_stopwords=True)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'unique_word_count': len(set(words)),
            'sentence_count': len(sentences),
            'avg_word_length': round(np.mean([len(w) for w in words]), 2) if words else 0,
            'avg_sentence_length': round(len(words) / len(sentences), 2) if sentences else 0,
            'lexical_diversity': round(len(set(words)) / len(words), 3) if words else 0,
            'content_word_ratio': round(len(words_no_stop) / len(words), 3) if words else 0
        }
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for processing
        Useful for transformer models with token limits
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def highlight_terms(self, text: str, terms: List[str]) -> str:
        """
        Highlight specific terms in text (useful for display)
        Returns text with **term** markdown
        """
        highlighted = text
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
        
        return highlighted


# Convenience functions for common operations

def quick_clean(text: str) -> str:
    """Quick text cleaning"""
    processor = TextProcessor()
    return processor.clean_text(text)

def get_keywords(text: str, n: int = 10) -> List[str]:
    """Get top N keywords from text"""
    processor = TextProcessor()
    keywords = processor.extract_keywords(text, top_n=n)
    return [kw for kw, _ in keywords]

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract domain entities"""
    processor = TextProcessor()
    return processor.extract_domain_entities(text)

def get_stats(text: str) -> Dict[str, any]:
    """Get text statistics"""
    processor = TextProcessor()
    return processor.get_text_statistics(text)


# Example usage and testing
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Test text
    sample_text = """
    This study investigated the effects of microgravity on human muscle tissue 
    during long-duration spaceflight aboard the ISS. Results showed significant 
    atrophy in lower limb muscles after 6 months in space. Gene expression analysis 
    revealed changes in 234 genes related to protein synthesis and degradation.
    """
    
    print("=== Text Processing Demo ===\n")
    
    print("1. Cleaned Text:")
    print(processor.clean_text(sample_text))
    
    print("\n2. Keywords:")
    keywords = processor.extract_keywords(sample_text, top_n=5)
    for word, freq in keywords:
        print(f"  - {word}: {freq}")
    
    print("\n3. Domain Entities:")
    entities = processor.extract_domain_entities(sample_text)
    for entity_type, found in entities.items():
        print(f"  - {entity_type}: {found}")
    
    print("\n4. Text Statistics:")
    stats = processor.get_text_statistics(sample_text)
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n5. Readability:")
    readability = processor.calculate_readability_score(sample_text)
    for key, value in readability.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Text Processor Ready!")