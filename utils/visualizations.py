import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import colorsys

class VisualizationEngine:
    """
    Advanced visualization engine for space biology data
    Generates publication-quality interactive visualizations
    """
    
    def __init__(self, theme='dark'):
        self.theme = theme
        self.colors = self._get_color_scheme()
    
    def _get_color_scheme(self):
        """Define color schemes for different visualizations"""
        if self.theme == 'dark':
            return {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'accent': '#f093fb',
                'success': '#4facfe',
                'warning': '#fa709a',
                'danger': '#ff6b6b',
                'background': '#1a1a2e',
                'text': '#ffffff',
                'grid': '#2d2d44'
            }
        else:
            return {
                'primary': '#5a67d8',
                'secondary': '#6b46c1',
                'accent': '#ed64a6',
                'success': '#48bb78',
                'warning': '#ed8936',
                'danger': '#f56565',
                'background': '#ffffff',
                'text': '#1a202c',
                'grid': '#e2e8f0'
            }
    
    def create_temporal_heatmap(self, papers):
        """
        Create temporal heatmap showing research intensity over time and topics
        """
        # Simulate data (replace with real extraction)
        years = list(range(2010, 2026))
        topics = ['Muscle Biology', 'Plant Sciences', 'Radiation', 
                 'Cardiovascular', 'Bone Biology', 'Immunology']
        
        # Create matrix
        np.random.seed(42)
        data_matrix = np.random.randint(5, 30, size=(len(topics), len(years)))
        
        # Add trend
        for i in range(len(topics)):
            trend = np.linspace(0, 10, len(years))
            data_matrix[i] = data_matrix[i] + trend.astype(int)
        
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=years,
            y=topics,
            colorscale=[
                [0, self.colors['background']],
                [0.3, self.colors['primary']],
                [0.6, self.colors['accent']],
                [1, self.colors['warning']]
            ],
            text=data_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                title="Publications",
                thickness=15,
                len=0.7
            )
        ))
        
        fig.update_layout(
            title="Research Intensity: Topics Over Time",
            xaxis_title="Year",
            yaxis_title="Research Area",
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=500,
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_network_graph_3d(self, graph, sample_size=50):
        """
        Create 3D interactive network graph
        """
        # Sample nodes if graph is too large
        if len(graph.nodes()) > sample_size:
            nodes = list(graph.nodes())[:sample_size]
            subgraph = graph.subgraph(nodes)
        else:
            subgraph = graph
        
        # Calculate 3D layout
        pos = nx.spring_layout(subgraph, dim=3, k=2, iterations=50)
        
        # Extract coordinates
        edge_x, edge_y, edge_z = [], [], []
        for edge in subgraph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.3)', width=2),
            hoverinfo='none',
            showlegend=False
        )
        
        # Node coordinates and properties
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_sizes = []
        node_colors = []
        
        type_colors = {
            'ORGANISM': self.colors['primary'],
            'ENVIRONMENT': self.colors['accent'],
            'EFFECT': self.colors['warning'],
            'BODY_SYSTEM': self.colors['success']
        }
        
        for node in subgraph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            # Node properties
            node_type = subgraph.nodes[node].get('type', 'UNKNOWN')
            degree = subgraph.degree(node)
            
            node_text.append(f"{node}<br>Type: {node_type}<br>Connections: {degree}")
            node_sizes.append(10 + degree * 2)
            node_colors.append(type_colors.get(node_type, self.colors['text']))
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=[str(n)[:15] for n in subgraph.nodes()],
            textposition="top center",
            textfont=dict(size=8, color=self.colors['text']),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color=self.colors['text'], width=0.5),
                opacity=0.8
            ),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="3D Knowledge Graph Network",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor=self.colors['background']
            ),
            showlegend=False,
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_sunburst_taxonomy(self, papers):
        """
        Create sunburst chart showing hierarchical taxonomy of research
        """
        # Create hierarchical data structure
        data = {
            'labels': ['Space Biology', 
                      'Human Biology', 'Plant Biology', 'Microbiology',
                      'Cardiovascular', 'Musculoskeletal', 'Neural',
                      'Growth', 'Stress Response',
                      'Bacteria', 'Fungi'],
            'parents': ['',
                       'Space Biology', 'Space Biology', 'Space Biology',
                       'Human Biology', 'Human Biology', 'Human Biology',
                       'Plant Biology', 'Plant Biology',
                       'Microbiology', 'Microbiology'],
            'values': [608,
                      250, 180, 178,
                      85, 95, 70,
                      90, 90,
                      98, 80]
        }
        
        fig = go.Figure(go.Sunburst(
            labels=data['labels'],
            parents=data['parents'],
            values=data['values'],
            branchvalues="total",
            marker=dict(
                colorscale='Purples',
                line=dict(color=self.colors['background'], width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Papers: %{value}<br>%{percentParent}',
            textfont=dict(size=12, color='white')
        ))
        
        fig.update_layout(
            title="Research Taxonomy Hierarchy",
            height=600,
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_gap_analysis_radar(self, gaps):
        """
        Create radar chart for knowledge gap analysis
        """
        categories = [gap['area'][:30] for gap in gaps[:8]]  # Top 8 gaps
        priorities = {
            'Critical': 3,
            'High': 2,
            'Medium': 1
        }
        
        values = [priorities.get(gap['priority'], 1) for gap in gaps[:8]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f"rgba(102, 126, 234, 0.3)",
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=8, color=self.colors['accent']),
            name='Priority Level'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 3],
                    ticktext=['', 'Medium', 'High', 'Critical'],
                    tickvals=[0, 1, 2, 3]
                ),
                bgcolor=self.colors['background']
            ),
            title="Knowledge Gap Priority Analysis",
            showlegend=False,
            height=500,
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_citation_flow_sankey(self, papers):
        """
        Create Sankey diagram showing flow from organisms to effects to missions
        """
        # Define flows (source -> target with value)
        flows = [
            ('Human', 'Microgravity Effects', 85),
            ('Human', 'Radiation Exposure', 45),
            ('Mouse', 'Microgravity Effects', 42),
            ('Mouse', 'Radiation Exposure', 38),
            ('Plant', 'Growth Changes', 35),
            ('Bacteria', 'Stress Response', 25),
            
            ('Microgravity Effects', 'ISS Experiments', 95),
            ('Microgravity Effects', 'Moon Missions', 32),
            ('Radiation Exposure', 'Mars Missions', 55),
            ('Radiation Exposure', 'ISS Experiments', 28),
            ('Growth Changes', 'Life Support Systems', 30),
            ('Stress Response', 'ISS Experiments', 22),
        ]
        
        # Extract unique nodes
        sources = []
        targets = []
        values = []
        
        all_nodes = set()
        for src, tgt, val in flows:
            all_nodes.add(src)
            all_nodes.add(tgt)
        
        node_list = list(all_nodes)
        node_dict = {node: idx for idx, node in enumerate(node_list)}
        
        for src, tgt, val in flows:
            sources.append(node_dict[src])
            targets.append(node_dict[tgt])
            values.append(val)
        
        # Assign colors
        colors = []
        for node in node_list:
            if node in ['Human', 'Mouse', 'Plant', 'Bacteria']:
                colors.append(self.colors['primary'])
            elif 'Effect' in node or 'Exposure' in node or 'Changes' in node or 'Response' in node:
                colors.append(self.colors['accent'])
            else:
                colors.append(self.colors['success'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color=self.colors['background'], width=2),
                label=node_list,
                color=colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(102, 126, 234, 0.3)'
            )
        )])
        
        fig.update_layout(
            title="Research Flow: Organisms → Effects → Applications",
            height=600,
            font=dict(size=12, color=self.colors['text']),
            paper_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_research_timeline(self, papers):
        """
        Create interactive timeline with milestone markers
        """
        # Simulate timeline data
        years = list(range(2010, 2026))
        publications = [20 + i*3 + np.random.randint(-5, 5) for i in range(len(years))]
        
        # Key milestones
        milestones = [
            {'year': 2011, 'event': 'ISS Life Sciences Lab Established', 'impact': 25},
            {'year': 2015, 'event': 'One-Year Mission Begins', 'impact': 40},
            {'year': 2018, 'event': 'Mars 2020 Prep Intensifies', 'impact': 35},
            {'year': 2020, 'event': 'COVID Impact on Research', 'impact': 30},
            {'year': 2024, 'event': 'Artemis Program Launch', 'impact': 55}
        ]
        
        fig = go.Figure()
        
        # Area chart for publications
        fig.add_trace(go.Scatter(
            x=years,
            y=publications,
            mode='lines',
            fill='tozeroy',
            fillcolor=f'rgba(102, 126, 234, 0.3)',
            line=dict(color=self.colors['primary'], width=3),
            name='Publications',
            hovertemplate='Year: %{x}<br>Publications: %{y}<extra></extra>'
        ))
        
        # Add milestone markers
        for milestone in milestones:
            fig.add_trace(go.Scatter(
                x=[milestone['year']],
                y=[milestone['impact']],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=self.colors['warning'],
                    line=dict(color=self.colors['text'], width=2),
                    symbol='star'
                ),
                text=[milestone['event']],
                textposition="top center",
                textfont=dict(size=9),
                showlegend=False,
                hovertemplate=f"<b>{milestone['event']}</b><br>Year: {milestone['year']}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Research Timeline with Key Milestones",
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=500,
            hovermode='x unified',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_collaborative_network(self, papers):
        """
        Create co-authorship/institution network visualization
        """
        # Simulate collaboration data
        institutions = [
            'NASA', 'MIT', 'Stanford', 'Harvard', 'ESA', 'JAXA',
            'UC Berkeley', 'Cambridge', 'Max Planck', 'CNES'
        ]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for inst in institutions:
            papers_count = np.random.randint(10, 50)
            G.add_node(inst, papers=papers_count, size=papers_count)
        
        # Add edges (collaborations)
        for i in range(len(institutions)):
            for j in range(i+1, len(institutions)):
                if np.random.random() > 0.6:  # 40% collaboration rate
                    collab_count = np.random.randint(1, 15)
                    G.add_edge(institutions[i], institutions[j], weight=collab_count)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight/2, color=self.colors['accent']),
                opacity=0.5,
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            papers = G.nodes[node]['papers']
            connections = G.degree(node)
            node_text.append(f"{node}<br>Papers: {papers}<br>Collaborations: {connections}")
            node_sizes.append(papers)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=self.colors['primary'],
                line=dict(color=self.colors['text'], width=2),
                sizemode='area',
                sizeref=2.*max(node_sizes)/(40.**2),
                sizemin=4
            ),
            text=list(G.nodes()),
            textposition="top center",
            textfont=dict(size=10, color=self.colors['text']),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title="Institutional Collaboration Network",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_organism_comparison_matrix(self, papers):
        """
        Create comparison matrix for different model organisms
        """
        organisms = ['Human', 'Mouse', 'Rat', 'Plant', 'Bacteria', 'Yeast']
        metrics = ['Publications', 'ISS Experiments', 'Genetic Tools', 
                  'Cost Efficiency', 'Translational Value']
        
        # Simulated scores (0-100)
        scores = np.array([
            [85, 95, 60, 40, 95],  # Human
            [75, 80, 90, 60, 80],  # Mouse
            [65, 70, 85, 65, 75],  # Rat
            [70, 75, 80, 85, 60],  # Plant
            [80, 65, 95, 95, 55],  # Bacteria
            [60, 55, 90, 90, 50]   # Yeast
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=scores,
            x=metrics,
            y=organisms,
            colorscale='Purples',
            text=scores,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='Organism: %{y}<br>Metric: %{x}<br>Score: %{z}<extra></extra>',
            colorbar=dict(title="Score", thickness=15)
        ))
        
        fig.update_layout(
            title="Model Organism Comparison Matrix",
            xaxis_title="Evaluation Metrics",
            yaxis_title="Model Organisms",
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=500
        )
        
        return fig
    
    def create_funding_allocation_treemap(self):
        """
        Create treemap showing research funding allocation
        """
        data = dict(
            labels=['Space Biology', 
                   'Human Research', 'Plant Biology', 'Microbiology', 'Physical Sciences',
                   'Cardiovascular', 'Bone & Muscle', 'Neuroscience',
                   'Crops', 'Algae',
                   'Extremophiles', 'Microbiome'],
            parents=['',
                    'Space Biology', 'Space Biology', 'Space Biology', 'Space Biology',
                    'Human Research', 'Human Research', 'Human Research',
                    'Plant Biology', 'Plant Biology',
                    'Microbiology', 'Microbiology'],
            values=[100,
                   45, 25, 18, 12,
                   20, 15, 10,
                   15, 10,
                   10, 8]
        )
        
        fig = go.Figure(go.Treemap(
            labels=data['labels'],
            parents=data['parents'],
            values=data['values'],
            textinfo="label+value+percent parent",
            marker=dict(
                colorscale='Purples',
                line=dict(width=2, color=self.colors['background'])
            ),
            hovertemplate='<b>%{label}</b><br>Funding: $%{value}M<br>%{percentParent}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Research Funding Allocation (Millions USD)",
            height=600,
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_multi_metric_dashboard(self, papers):
        """
        Create comprehensive multi-metric dashboard with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Publications Over Time', 'Top Research Areas',
                          'Organism Distribution', 'Collaboration Index'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'indicator'}]]
        )
        
        # 1. Time series
        years = list(range(2010, 2026))
        pubs = [20 + i*3 for i in range(len(years))]
        fig.add_trace(
            go.Scatter(x=years, y=pubs, mode='lines+markers',
                      line=dict(color=self.colors['primary'], width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # 2. Bar chart
        topics = ['Muscle', 'Plant', 'Radiation', 'Cardio', 'Bone']
        counts = [45, 38, 29, 33, 27]
        fig.add_trace(
            go.Bar(x=topics, y=counts, marker_color=self.colors['accent']),
            row=1, col=2
        )
        
        # 3. Pie chart
        organisms = ['Human', 'Mouse', 'Plant', 'Bacteria', 'Other']
        percentages = [35, 25, 20, 12, 8]
        fig.add_trace(
            go.Pie(labels=organisms, values=percentages,
                  marker=dict(colors=px.colors.sequential.Purples)),
            row=2, col=1
        )
        
        # 4. Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=85,
                delta={'reference': 70, 'increasing': {'color': self.colors['success']}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': self.colors['background']},
                        {'range': [50, 75], 'color': self.colors['grid']},
                    ],
                    'threshold': {
                        'line': {'color': self.colors['warning'], 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title={'text': "Research Impact Score"}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            title_text="Space Biology Research Dashboard",
            title_font_size=20
        )
        
        return fig
    
    def create_word_cloud_data(self, papers, top_n=50):
        """
        Extract word frequency data for word cloud visualization
        (Returns data structure for frontend word cloud library)
        """
        from collections import Counter
        import re
        
        # Combine all text
        all_text = ""
        for paper in papers:
            all_text += " " + paper.get('title', '')
            all_text += " " + paper.get('abstract', '')[:500]
        
        # Tokenize and filter
        words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
        
        # Remove common words
        stop_words = set(['abstract', 'study', 'research', 'using', 'results', 
                         'showed', 'found', 'observed', 'data', 'analysis'])
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        word_freq = Counter(words).most_common(top_n)
        
        return [{'text': word, 'value': freq} for word, freq in word_freq]


# Example usage function for Streamlit integration
def render_all_visualizations(papers, graph=None, gaps=None):
    """
    Helper function to render all visualizations in Streamlit
    Usage: render_all_visualizations(papers, graph, gaps)
    """
    import streamlit as st
    
    viz_engine = VisualizationEngine(theme='dark')
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analytics", "🕸️ Networks", "📈 Trends", "🎯 Comparisons"
    ])
    
    with tab1:
        st.plotly_chart(viz_engine.create_multi_metric_dashboard(papers), 
                       use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz_engine.create_sunburst_taxonomy(papers),
                           use_container_width=True)
        with col2:
            st.plotly_chart(viz_engine.create_funding_allocation_treemap(),
                           use_container_width=True)
    
    with tab2:
        if graph:
            st.plotly_chart(viz_engine.create_network_graph_3d(graph),
                           use_container_width=True)
        
        st.plotly_chart(viz_engine.create_collaborative_network(papers),
                       use_container_width=True)
        
        st.plotly_chart(viz_engine.create_citation_flow_sankey(papers),
                       use_container_width=True)
    
    with tab3:
        st.plotly_chart(viz_engine.create_research_timeline(papers),
                       use_container_width=True)
        
        st.plotly_chart(viz_engine.create_temporal_heatmap(papers),
                       use_container_width=True)
    
    with tab4:
        st.plotly_chart(viz_engine.create_organism_comparison_matrix(papers),
                       use_container_width=True)
        
        if gaps:
            st.plotly_chart(viz_engine.create_gap_analysis_radar(gaps),
                           use_container_width=True)


if __name__ == "__main__":
    # Test visualizations
    print("Visualization engine ready!")
    print("Import this module in your Streamlit app:")
    print("from utils.visualizations import render_all_visualizations")
    print("render_all_visualizations(papers, graph, gaps)")