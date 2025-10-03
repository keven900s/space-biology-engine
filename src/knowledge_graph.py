# src/knowledge_graph.py
# Advanced Knowledge Graph Construction for Space Biology

import networkx as nx
import spacy
from collections import defaultdict
import json
from pathlib import Path

class KnowledgeGraphBuilder:
    """
    Build a comprehensive knowledge graph from space biology papers
    connecting: Organisms -> Experiments -> Environments -> Effects -> Missions
    """
    
    def __init__(self, nlp_model=None):
        self.graph = nx.MultiDiGraph()
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        
        # Define ontology/schema
        self.entity_types = {
            'ORGANISM': ['human', 'mouse', 'rat', 'plant', 'arabidopsis', 'bacteria', 
                        'cell', 'yeast', 'drosophila', 'c. elegans'],
            'ENVIRONMENT': ['microgravity', 'space', 'iss', 'radiation', 'mars', 
                           'lunar', 'analog', 'simulated', 'parabolic flight'],
            'BODY_SYSTEM': ['cardiovascular', 'musculoskeletal', 'nervous', 'immune',
                           'digestive', 'respiratory', 'bone', 'muscle'],
            'BIOLOGICAL_PROCESS': ['growth', 'development', 'metabolism', 'gene expression',
                                  'cell division', 'apoptosis', 'differentiation'],
            'EFFECT': ['atrophy', 'loss', 'damage', 'adaptation', 'change', 'increase',
                      'decrease', 'dysfunction', 'recovery'],
            'MISSION': ['apollo', 'shuttle', 'iss', 'spacex', 'artemis', 'mars', 'skylab']
        }
        
        # Relation types
        self.relation_types = [
            'STUDIED_IN',      # Organism -> Environment
            'AFFECTED_BY',     # Body System -> Environment
            'CAUSES',          # Environment -> Effect
            'INVOLVES',        # Experiment -> Organism
            'TARGETS',         # Research -> Body System
            'SUPPORTS',        # Finding -> Mission
            'RELATED_TO'       # General connection
        ]
    
    def extract_entities_from_text(self, text, paper_id=None):
        """Extract biological entities using pattern matching and NER"""
        doc = self.nlp(text[:100000])  # Spacy token limit
        
        entities = defaultdict(list)
        
        # Pattern-based extraction
        text_lower = text.lower()
        for entity_type, keywords in self.entity_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities[entity_type].append({
                        'text': keyword,
                        'type': entity_type,
                        'paper_id': paper_id
                    })
        
        # NER-based extraction
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE'] and any(mission in ent.text.lower() 
                                                     for mission in self.entity_types['MISSION']):
                entities['MISSION'].append({
                    'text': ent.text,
                    'type': 'MISSION',
                    'paper_id': paper_id
                })
        
        # Deduplicate
        for entity_type in entities:
            seen = set()
            unique = []
            for entity in entities[entity_type]:
                if entity['text'] not in seen:
                    seen.add(entity['text'])
                    unique.append(entity)
            entities[entity_type] = unique
        
        return entities
    
    def extract_relations(self, text, entities):
        """Extract relationships between entities using dependency parsing"""
        doc = self.nlp(text[:50000])
        relations = []
        
        # Simple rule-based relation extraction
        # Pattern: "X affects Y", "X causes Y", "X studied in Y"
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Find entity pairs in sentence
            sent_entities = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity['text'].lower() in sent_text:
                        sent_entities.append((entity['text'], entity_type))
            
            # Extract relations based on patterns
            if len(sent_entities) >= 2:
                for i, (ent1, type1) in enumerate(sent_entities):
                    for ent2, type2 in sent_entities[i+1:]:
                        relation = self._determine_relation(sent_text, type1, type2)
                        if relation:
                            relations.append({
                                'source': ent1,
                                'target': ent2,
                                'relation': relation,
                                'context': sent.text[:200]
                            })
        
        return relations
    
    def _determine_relation(self, sentence, type1, type2):
        """Determine relation type based on entity types and sentence content"""
        patterns = {
            'STUDIED_IN': ['studied in', 'examined in', 'tested in', 'exposed to'],
            'AFFECTED_BY': ['affected by', 'influenced by', 'impacted by', 'altered by'],
            'CAUSES': ['causes', 'leads to', 'results in', 'induces'],
            'INVOLVES': ['involves', 'includes', 'uses', 'employs'],
        }
        
        for relation, keywords in patterns.items():
            if any(kw in sentence for kw in keywords):
                return relation
        
        # Default based on entity types
        if type1 == 'ORGANISM' and type2 == 'ENVIRONMENT':
            return 'STUDIED_IN'
        elif type1 == 'ENVIRONMENT' and type2 == 'EFFECT':
            return 'CAUSES'
        elif type1 == 'BODY_SYSTEM' and type2 == 'EFFECT':
            return 'SHOWS'
        
        return 'RELATED_TO'
    
    def build_graph_from_papers(self, papers):
        """Build complete knowledge graph from all papers"""
        print(f"Building knowledge graph from {len(papers)} papers...")
        
        for paper in papers:
            paper_id = paper.get('id', paper.get('title', 'unknown'))
            
            # Extract from abstract and conclusion (most information-dense)
            text = ' '.join([
                paper.get('title', ''),
                paper.get('abstract', ''),
                paper.get('conclusion', '')
            ])
            
            if not text.strip():
                continue
            
            # Extract entities
            entities = self.extract_entities_from_text(text, paper_id)
            
            # Add nodes
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    self.graph.add_node(
                        entity['text'],
                        type=entity_type,
                        papers=[paper_id],
                        label=entity['text']
                    )
            
            # Extract and add relations
            relations = self.extract_relations(text, entities)
            for rel in relations:
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    relation=rel['relation'],
                    paper_id=paper_id,
                    context=rel['context']
                )
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def get_subgraph(self, central_node, depth=2):
        """Extract subgraph around a central node"""
        if central_node not in self.graph:
            return nx.Graph()
        
        # BFS to get neighbors within depth
        nodes = {central_node}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)
        
        return self.graph.subgraph(nodes).copy()
    
    def find_paths(self, source, target, max_length=4):
        """Find all paths between two concepts"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths[:10]  # Return top 10
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_most_connected_nodes(self, top_k=20):
        """Get hub nodes (most connected concepts)"""
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def get_node_info(self, node_name):
        """Get detailed information about a node"""
        if node_name not in self.graph:
            return None
        
        node_data = self.graph.nodes[node_name]
        
        # Get connections
        neighbors = list(self.graph.neighbors(node_name))
        in_edges = self.graph.in_edges(node_name, data=True)
        out_edges = self.graph.out_edges(node_name, data=True)
        
        return {
            'name': node_name,
            'type': node_data.get('type', 'UNKNOWN'),
            'papers': node_data.get('papers', []),
            'degree': self.graph.degree(node_name),
            'neighbors': neighbors[:20],
            'incoming_relations': [(e[0], e[2].get('relation', 'RELATED')) for e in in_edges][:10],
            'outgoing_relations': [(e[1], e[2].get('relation', 'RELATED')) for e in out_edges][:10]
        }
    
    def export_graph(self, filepath):
        """Export graph for visualization"""
        # Convert to JSON for D3.js
        data = {
            'nodes': [
                {
                    'id': node,
                    'type': self.graph.nodes[node].get('type', 'UNKNOWN'),
                    'papers': len(self.graph.nodes[node].get('papers', [])),
                    'degree': self.graph.degree(node)
                }
                for node in self.graph.nodes()
            ],
            'links': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'relation': self.graph.edges[edge].get('relation', 'RELATED'),
                    'weight': 1
                }
                for edge in self.graph.edges()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Graph exported to {filepath}")
    
    def get_research_gaps(self):
        """Identify under-connected areas in the graph"""
        gaps = []
        
        # Find low-degree nodes (under-researched)
        for node, degree in self.graph.degree():
            if degree < 3:  # Less than 3 connections
                node_type = self.graph.nodes[node].get('type', 'UNKNOWN')
                paper_count = len(self.graph.nodes[node].get('papers', []))
                
                if paper_count > 0 and paper_count < 5:
                    gaps.append({
                        'concept': node,
                        'type': node_type,
                        'connections': degree,
                        'papers': paper_count,
                        'priority': 'HIGH' if paper_count <= 2 else 'MEDIUM'
                    })
        
        return sorted(gaps, key=lambda x: x['papers'])[:20]
    
    def get_cross_domain_insights(self):
        """Find unexpected connections between different domains"""
        insights = []
        
        # Look for connections between distant entity types
        for node1 in list(self.graph.nodes())[:100]:  # Sample for performance
            type1 = self.graph.nodes[node1].get('type')
            
            for node2 in self.graph.neighbors(node1):
                type2 = self.graph.nodes[node2].get('type')
                
                # Interesting if connecting different high-level categories
                if type1 and type2 and type1 != type2:
                    if (type1, type2) in [('ORGANISM', 'MISSION'), ('EFFECT', 'MISSION')]:
                        insights.append({
                            'connection': f"{node1} ↔ {node2}",
                            'types': f"{type1} → {type2}",
                            'papers': len(set(self.graph.nodes[node1].get('papers', [])) & 
                                        set(self.graph.nodes[node2].get('papers', [])))
                        })
        
        return sorted(insights, key=lambda x: x['papers'], reverse=True)[:15]
    
    def generate_cypher_queries(self):
        """Generate Neo4j Cypher queries for advanced graph database"""
        queries = []
        
        # Create nodes
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'CONCEPT')
            papers = data.get('papers', [])
            query = f"""
            CREATE (n:{node_type} {{
                name: "{node}",
                paper_count: {len(papers)},
                papers: {papers[:5]}
            }})
            """
            queries.append(query)
        
        # Create relationships
        for edge in self.graph.edges(data=True):
            relation = edge[2].get('relation', 'RELATED_TO')
            query = f"""
            MATCH (a {{name: "{edge[0]}"}}), (b {{name: "{edge[1]}"}})
            CREATE (a)-[:{relation}]->(b)
            """
            queries.append(query)
        
        return queries