from collections import Counter
import pandas as pd

class InsightsEngine:
    def __init__(self, papers):
        self.papers = papers
    
    def identify_knowledge_gaps(self):
        """Identify under-researched areas"""
        # Extract all topics/entities
        all_topics = []
        for paper in self.papers:
            entities = paper.get('entities', {})
            all_topics.extend(entities.get('organisms', []))
            all_topics.extend(entities.get('environments', []))
        
        topic_counts = Counter(all_topics)
        
        # Find rare topics (potential gaps)
        gaps = []
        for topic, count in topic_counts.items():
            if count < 5 and count > 0:  # Between 1-4 papers
                priority = 'Critical' if count <= 2 else 'High'
                gaps.append({
                    'area': topic,
                    'priority': priority,
                    'study_count': count
                })
        
        return sorted(gaps, key=lambda x: x['study_count'])[:15]
    
    def analyze_trends(self):
        """Analyze research trends over time"""
        # Extract years from papers (you'd parse this from metadata)
        trends = {
            'total_papers_per_year': {},
            'trending_topics': [],
            'emerging_areas': []
        }
        return trends
    
    def generate_mission_insights(self, mission_type='mars'):
        """Generate insights for specific missions"""
        relevant_keywords = {
            'mars': ['radiation', 'long-duration', 'isolation', 'Mars analog'],
            'moon': ['lunar', 'partial gravity', 'dust exposure']
        }
        
        keywords = relevant_keywords.get(mission_type, [])
        relevant_papers = []
        
        for paper in self.papers:
            text = paper.get('abstract', '') + paper.get('conclusion', '')
            if any(kw in text.lower() for kw in keywords):
                relevant_papers.append(paper)
        
        return {
            'mission': mission_type,
            'relevant_studies': len(relevant_papers),
            'top_concerns': ['Radiation exposure', 'Muscle atrophy', 'Immune dysfunction'],
            'recommendations': [
                'Implement countermeasures for bone density loss',
                'Monitor cardiovascular adaptation',
                'Research plant-based life support systems'
            ]
        }