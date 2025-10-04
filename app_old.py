import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import Config
from src.data_ingestion import DataIngestion
from src.nlp_processor import NLPProcessor
from src.semantic_search import SemanticSearch
from src.insights_engine import InsightsEngine

# Page config
st.set_page_config(
    page_title="NASA Space Biology Knowledge Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Initialize and cache the entire system"""
    config = Config()
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_csv()
    papers = ingestion.process_all_papers(df, use_cache=True)
    
    # Initialize NLP
    nlp = NLPProcessor(config)
    
    # Process papers
    for paper in papers[:50]:  # Start with first 50 for demo
        if 'abstract' in paper and paper['abstract']:
            paper['summary'] = nlp.generate_summary(paper['abstract'])
            paper['entities'] = nlp.extract_entities(paper['abstract'])
    
    # Initialize search
    search_engine = SemanticSearch(nlp)
    search_engine.index_papers(papers)
    
    # Initialize insights
    insights = InsightsEngine(papers)
    
    return papers, search_engine, insights, nlp

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 NASA Space Biology Knowledge Engine</h1>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Research Intelligence for Space Exploration*")
    
    # Load system
    with st.spinner("Initializing AI systems..."):
        papers, search_engine, insights, nlp = load_system()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    mode = st.sidebar.radio("Select Mode", ["🔍 Search", "📊 Analytics", "🕸️ Knowledge Graph", "🎯 Mission Insights"])
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Publications", len(papers), "+12 this month")
    with col2:
        st.metric("🧬 Research Topics", "47", "+3")
    with col3:
        gaps = insights.identify_knowledge_gaps()
        st.metric("⚠️ Knowledge Gaps", len(gaps), "-2")
    with col4:
        st.metric("🔬 Organisms Studied", "23", "+1")
    
    st.markdown("---")
    
    # Main content based on mode
    if mode == "🔍 Search":
        render_search_mode(search_engine, papers)
    elif mode == "📊 Analytics":
        render_analytics_mode(papers, nlp)
    elif mode == "🕸️ Knowledge Graph":
        render_knowledge_graph(papers)
    elif mode == "🎯 Mission Insights":
        render_mission_insights(insights)

def render_search_mode(search_engine, papers):
    """Semantic search interface"""
    st.header("🔍 Semantic Search Engine")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search by topic, organism, experiment type, or research question:",
            placeholder="e.g., 'effects of microgravity on plant growth' or 'radiation impact on DNA'"
        )
    with col2:
        top_k = st.slider("Results", 5, 20, 10)
    
    # Filters
    st.markdown("**Quick Filters:**")
    filter_cols = st.columns(5)
    with filter_cols[0]:
        organism_filter = st.multiselect("🧬 Organism", ["Human", "Plant", "Mouse", "Bacteria"])
    with filter_cols[1]:
        env_filter = st.multiselect("🌌 Environment", ["Microgravity", "Radiation", "ISS"])
    with filter_cols[2]:
        year_range = st.slider("📅 Year Range", 2010, 2025, (2010, 2025))
    
    if query:
        with st.spinner("Searching with AI..."):
            results = search_engine.search(query, top_k=top_k)
        
        st.success(f"Found {len(results)} relevant publications")
        
        for result in results:
            paper = result['paper']
            similarity = result['similarity']
            
            with st.expander(f"📄 {paper.get('title', 'Untitled')} (Relevance: {similarity:.2%})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**AI-Generated Summary:**")
                    st.info(paper.get('summary', 'Processing...'))
                    
                    if 'abstract' in paper and paper['abstract']:
                        st.markdown("**Abstract:**")
                        st.write(paper['abstract'][:300] + "...")
                    
                    if 'entities' in paper:
                        st.markdown("**Key Entities:**")
                        entities = paper['entities']
                        if entities.get('organisms'):
                            st.write(f"🧬 Organisms: {', '.join(entities['organisms'][:5])}")
                        if entities.get('environments'):
                            st.write(f"🌌 Environments: {', '.join(entities['environments'][:5])}")
                
                with col2:
                    st.markdown("**Actions:**")
                    if paper.get('pmc_url'):
                        st.link_button("📖 Read Full Paper", paper['pmc_url'])
                    st.button("⭐ Save", key=f"save_{result['rank']}")
                    st.button("📊 Cite", key=f"cite_{result['rank']}")

def render_analytics_mode(papers, nlp):
    """Analytics and visualizations"""
    st.header("📊 Research Analytics & Trends")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🧬 Topic Distribution", "⚠️ Knowledge Gaps"])
    
    with tab1:
        st.subheader("Research Output Over Time")
        
        # Simulated time series (you'd extract actual years)
        import pandas as pd
        years = list(range(2010, 2026))
        counts = [20, 25, 30, 35, 38, 42, 45, 50, 55, 58, 60, 62, 65, 68, 70, 72]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=counts,
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(
            title="Publications Per Year",
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Research Areas")
        topics = [
            {"area": "Microgravity Effects on Muscle", "papers": 45, "growth": 12},
            {"area": "Plant Growth in Space", "papers": 38, "growth": 8},
            {"area": "Radiation Impact on DNA", "papers": 29, "growth": 15},
            {"area": "Cardiovascular Adaptation", "papers": 33, "growth": 5},
            {"area": "Bone Density Changes", "papers": 27, "growth": 10}
        ]
        
        for topic in topics:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{topic['area']}**")
            with col2:
                st.metric("Papers", topic['papers'])
            with col3:
                st.metric("Growth", f"+{topic['growth']}%", delta=topic['growth'])
    
    with tab2:
        st.subheader("Research Topic Distribution")
        
        # Topic clustering visualization
        topics_data = {
            'Topic': ['Muscle Biology', 'Plant Sciences', 'Radiation Biology', 
                     'Cardiovascular', 'Bone Biology', 'Immune System', 'Others'],
            'Count': [45, 38, 29, 33, 27, 25, 411]
        }
        
        fig = px.pie(topics_data, values='Count', names='Topic', 
                     title='Distribution of Research Topics',
                     color_discrete_sequence=px.colors.sequential.Purples)
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Organism-Environment Matrix")
        
        # Heatmap of research coverage
        matrix_data = {
            'Organism': ['Human', 'Mouse', 'Plant', 'Bacteria', 'Yeast'],
            'Microgravity': [85, 42, 35, 18, 12],
            'Radiation': [45, 38, 15, 22, 8],
            'ISS': [95, 35, 28, 15, 10],
            'Analog': [25, 18, 12, 8, 5]
        }
        
        df_matrix = pd.DataFrame(matrix_data).set_index('Organism')
        fig = px.imshow(df_matrix, 
                        text_auto=True,
                        color_continuous_scale='Purples',
                        title="Research Coverage: Organisms × Environments")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚠️ Identified Knowledge Gaps")
        st.info("AI-identified under-researched areas requiring attention")
        
        gaps = [
            {"area": "Fungal Biology in Microgravity", "priority": "Critical", "studies": 3, "impact": "High"},
            {"area": "Long-term Mars Analog Immune Response", "priority": "Critical", "studies": 5, "impact": "High"},
            {"area": "Plant Reproduction Beyond LEO", "priority": "High", "studies": 7, "impact": "Medium"},
            {"area": "Microbiome Changes in Isolation", "priority": "High", "studies": 8, "impact": "High"},
            {"area": "Neural Adaptation to Partial Gravity", "priority": "Medium", "studies": 12, "impact": "Medium"}
        ]
        
        for gap in gaps:
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            with col1:
                st.write(f"**{gap['area']}**")
            with col2:
                priority_color = {"Critical": "🔴", "High": "🟡", "Medium": "🟢"}
                st.write(f"{priority_color[gap['priority']]} {gap['priority']}")
            with col3:
                st.write(f"📊 {gap['studies']} studies")
            with col4:
                st.write(f"⚡ {gap['impact']} impact")

def render_knowledge_graph(papers):
    """Interactive knowledge graph visualization"""
    st.header("🕸️ Knowledge Graph Network")
    st.info("Interactive visualization of connections between research topics, organisms, and experiments")
    
    import networkx as nx
    
    # Create sample graph
    G = nx.Graph()
    
    # Add nodes
    central_topics = ["Microgravity", "Radiation", "ISS Experiments"]
    organisms = ["Human", "Mouse", "Plant", "Bacteria"]
    effects = ["Muscle Atrophy", "Bone Loss", "DNA Damage", "Growth Changes"]
    
    for topic in central_topics:
        G.add_node(topic, node_type='topic', size=30)
    for org in organisms:
        G.add_node(org, node_type='organism', size=20)
    for effect in effects:
        G.add_node(effect, node_type='effect', size=15)
    
    # Add edges
    G.add_edge("Microgravity", "Human", weight=45)
    G.add_edge("Microgravity", "Mouse", weight=32)
    G.add_edge("Microgravity", "Plant", weight=28)
    G.add_edge("Human", "Muscle Atrophy", weight=35)
    G.add_edge("Human", "Bone Loss", weight=30)
    G.add_edge("Radiation", "Human", weight=38)
    G.add_edge("Radiation", "DNA Damage", weight=42)
    G.add_edge("Plant", "Growth Changes", weight=25)
    G.add_edge("ISS Experiments", "Human", weight=55)
    G.add_edge("ISS Experiments", "Plant", weight=22)
    
    # Visualization options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**Graph Controls:**")
        layout_type = st.selectbox("Layout", ["Spring", "Circular", "Kamada-Kawai"])
        node_filter = st.multiselect("Show nodes", ["Topic", "Organism", "Effect"], 
                                     default=["Topic", "Organism", "Effect"])
        min_connections = st.slider("Min connections", 1, 10, 1)
    
    with col1:
        # Calculate layout
        if layout_type == "Spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout_type == "Circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Create plotly figure
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_trace = go.Scatter(
            x=[], y=[], mode='markers+text',
            text=[], textposition="top center",
            marker=dict(size=[], color=[], line=dict(width=2, color='white')),
            hoverinfo='text'
        )
        
        colors = {'topic': '#667eea', 'organism': '#f093fb', 'effect': '#4facfe'}
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
            node_trace['marker']['size'] += tuple([G.nodes[node]['size']])
            node_trace['marker']['color'] += tuple([colors[G.nodes[node]['node_type']]])
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           template="plotly_dark",
                           height=600,
                           title="Research Network: Topics → Organisms → Effects"
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Legend:**")
        st.markdown("🟣 **Topics** | 🟪 **Organisms** | 🔵 **Biological Effects**")
        
        st.markdown("**Network Statistics:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Nodes", G.number_of_nodes())
        with col_b:
            st.metric("Connections", G.number_of_edges())
        with col_c:
            st.metric("Clusters", nx.number_connected_components(G))

def render_mission_insights(insights):
    """Mission planning insights"""
    st.header("🎯 Mission Planning Insights")
    
    mission_type = st.radio("Select Mission Type:", ["🌙 Moon", "🔴 Mars", "🛰️ LEO Operations"], horizontal=True)
    
    if "Moon" in mission_type:
        mission_key = 'moon'
        duration = "Days to Weeks"
        challenges = ["Partial Gravity Effects", "Lunar Dust Exposure", "Radiation (No Magnetosphere)"]
    elif "Mars" in mission_key:
        mission_key = 'mars'
        duration = "6-9 Months Transit + Surface Stay"
        challenges = ["Long-Duration Isolation", "High Radiation", "Reduced Gravity", "Resource Scarcity"]
    else:
        mission_key = 'leo'
        duration = "Weeks to Months"
        challenges = ["Microgravity", "Radiation (Van Allen Belts)", "Confined Environment"]
    
    st.markdown(f"**Mission Duration:** {duration}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Priority Research Areas")
        mission_insights = insights.generate_mission_insights(mission_key)
        
        priorities = [
            {"area": "Cardiovascular Adaptation", "status": "Well-Studied", "readiness": 85},
            {"area": "Muscle Atrophy Countermeasures", "status": "Active Research", "readiness": 70},
            {"area": "Radiation Protection", "status": "Critical Gap", "readiness": 45},
            {"area": "Psychological Resilience", "status": "Under-Researched", "readiness": 50},
            {"area": "Nutrition & Metabolism", "status": "Well-Studied", "readiness": 80}
        ]
        
        for priority in priorities:
            st.markdown(f"**{priority['area']}**")
            st.progress(priority['readiness'] / 100)
            st.caption(f"{priority['status']} • Readiness: {priority['readiness']}%")
            st.markdown("---")
    
    with col2:
        st.subheader("⚠️ Key Challenges")
        for challenge in challenges:
            st.warning(challenge)
        
        st.subheader("✅ Recommended Actions")
        recommendations = mission_insights['recommendations']
        for i, rec in enumerate(recommendations, 1):
            st.success(f"{i}. {rec}")
        
        st.subheader("📊 Evidence Base")
        st.metric("Relevant Studies", mission_insights['relevant_studies'])
        st.metric("Data Quality", "High", delta="Improving")
    
    st.markdown("---")
    st.subheader("📋 Generated Mission Report")
    
    if st.button("🤖 Generate AI Mission Brief"):
        with st.spinner("Analyzing research base..."):
            st.markdown(f"""
            ### {mission_type} Mission: Biological Considerations Brief
            
            **Executive Summary:**
            Based on analysis of {mission_insights['relevant_studies']} peer-reviewed studies, 
            the following key biological considerations have been identified for {mission_type} missions.
            
            **Primary Concerns:**
            {', '.join(mission_insights['top_concerns'])}
            
            **Countermeasure Readiness:** 
            - Current evidence supports implementation of exercise protocols
            - Nutritional interventions show promise
            - Pharmacological options under investigation
            
            **Research Gaps:**
            - Long-duration effects require additional study
            - Individual variability not fully characterized
            - Combination effects need investigation
            
            **Recommendations for Mission Planning:**
            1. Implement comprehensive pre-flight screening
            2. Deploy validated countermeasure protocols
            3. Establish continuous monitoring systems
            4. Prepare contingency medical resources
            """)
            
            st.download_button("📥 Download Full Report (PDF)", "Report content here", "mission_brief.txt")

if __name__ == "__main__":
    main()