import streamlit as st
import argparse
from GraphRAG import GraphRAG
import time
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GraphRAG Research Assistant 🧠",
    page_icon="🧠",
    layout="wide"
)

#
#ADDITIONAL FEATURES (compared to main.py):
#- This caches the GraphRAG object to avoid reinitializing it on refreshing (it takes a looooong time).
#- I also added a display_network_graph function to display an interactive community graph within the app.
#- I can also interactively change the configuration of the GraphRAG object using the sidebar.


# Gen AI citation:
# - I used Copilot to help me turn main.py into an Streamlit application.

def main():
    if 'config' not in st.session_state:
        st.session_state.config = {
            'json_path': "datasets/arxiv_cs_metadata.json",
            'nrows': 5,
            'database': "neo4j",
            'llm': "qwen2.5",
            'embed_model': "avsolatorio/GIST-all-MiniLM-L6-v2"
        }

    if 'graph_rag' not in st.session_state:
        st.session_state.graph_rag = None
    
    st.title("Research Buddy 🧠")
    st.subheader("A GraphRAG Research Assistance Platform")
    
    # Sidebar for configurations
    with st.sidebar:
        st.header("Setup")
        new_json_path = st.text_input(
            "JSON Path",
            value=st.session_state.config['json_path'],
            help="Path to JSON file containing papers data"
        )
        new_nrows = st.number_input(
            "Number of Rows",
            min_value=1,
            value=st.session_state.config['nrows'],
            help="Number of rows to read from JSON"
        )
        new_database = st.selectbox(
            "Database",
            options=st.session_state.config['database'],
            help="Name of the neo4j database"
        )
        new_llm = st.text_input(
            "Ollama LLM Model",
            value=st.session_state.config['llm'],
            help="Ollama Model name"
        )
        new_embed_model = st.text_input(
            "HuggingFace Embedding Model",
            value=st.session_state.config['embed_model'],
            help="HuggingFace Embedding Model Name"
        )
        
        config_changed = (
            new_json_path != st.session_state.config['json_path'] or
            new_nrows != st.session_state.config['nrows'] or
            new_database != st.session_state.config['database'] or
            new_llm != st.session_state.config['llm'] or
            new_embed_model != st.session_state.config['embed_model']
        )
        
        if config_changed:
            st.warning("Config changed. Click 'Initialize GraphRAG' to apply.")
        
        initialize = st.button("Initialize GraphRAG")
        
        if initialize or (st.session_state.graph_rag is None):
            st.session_state.config.update({
                'json_path': new_json_path,
                'nrows': new_nrows,
                'database': new_database,
                'llm': new_llm,
                'embed_model': new_embed_model
            })

            st.session_state.graph_rag = get_graphrag(
                st.session_state.config['json_path'],
                st.session_state.config['nrows'],
                st.session_state.config['database'],
                st.session_state.config['llm'],
                st.session_state.config['embed_model']
            )
            st.success("GraphRAG initialized successfully!")
    
    if st.session_state.graph_rag is not None:
        st.markdown("---")
            
        with st.expander("View Community Graph"):
            display_network_graph("community_graph.html")

        query = st.text_area(
            "Enter your query:",
            height=100,
        )
        
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing your query..."):
                    response = st.session_state.graph_rag.query(query)
                st.markdown("### Response")
                st.markdown(response)
                st.caption(f"Query processed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("Please enter a query first.")
    else:
        st.info("Please initialize GraphRAG using the sidebar controls first.")

@st.cache_resource
def get_graphrag(json_path, nrows, database, llm, embed_model):
    with st.spinner('Setting up our GraphRAG. This might take a while...'):
        graph_rag = GraphRAG(
            json_path=json_path,
            nrows=nrows,
            database=database,
            llm=llm,
            embed_model=embed_model
        )
    return graph_rag

def display_network_graph(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" />
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"></script>
                <link href="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/css/tom-select.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/js/tom-select.complete.min.js"></script>
            </head>
            <body>
                {html_content}
            </body>
        </html>
        """,
        height=800,
        width=None,
        scrolling=True 
    )
if __name__ == "__main__":
    main()