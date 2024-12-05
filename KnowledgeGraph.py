import pandas as pd
from typing import List, Optional
from llama_index.core import Document, PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.schema import MetadataMode
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dataclasses import dataclass

class Neo4jConfig:
    def __init__(
        self,
        username: str,
        password: str,
        url: str = "bolt://localhost:7687",
        database: str = "arxivCs"
    ):
        self.username = username
        self.password = password
        self.url = url
        self.database = database

class ExtractorConfig:
    def __init__(
        self,
        max_triplets_per_chunk: int = 20,
        num_workers: int = 4,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relation_types: Optional[List[str]] = None,
        allowed_relation_props: List[str] = [],
        allowed_entity_props: List[str] = []
    ):
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.num_workers = num_workers
        self.allowed_entity_types = allowed_entity_types
        self.allowed_relation_types = allowed_relation_types
        self.allowed_relation_props = allowed_relation_props
        self.allowed_entity_props = allowed_entity_props

class LlamaIndexKGBuilder:
    def __init__(
        self,
        neo4j_config: Neo4jConfig,
        llm_model: str = "mistral",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        request_timeout: float = 300.0,
        extractor_config: Optional[ExtractorConfig] = None
    ):
        """
        Initialize the Knowledge Graph Builder.
        
        Args:
            neo4j_config: Neo4j connection configuration
            llm_model: Name of the Ollama model to use
            request_timeout: Timeout for LLM requests in seconds
            extractor_config: Configuration for the knowledge graph extractor
        """
        self.neo4j_config = neo4j_config
        self.llm = Ollama(model=llm_model, request_timeout=request_timeout)
        self.embed_model = HuggingFaceEmbedding(embed_model)

        self.extractor_config = extractor_config or ExtractorConfig()
        
        # Initialize Neo4j graph store
        self.graph_store = Neo4jPropertyGraphStore(
            username=neo4j_config.username,
            password=neo4j_config.password,
            url=neo4j_config.url,
            database=neo4j_config.database
        )
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            property_graph_store=self.graph_store
        )
        
        # Initialize knowledge graph extractor
        self.kg_extractor = DynamicLLMPathExtractor(
            llm=self.llm,
            max_triplets_per_chunk=self.extractor_config.max_triplets_per_chunk,
            num_workers=self.extractor_config.num_workers,
            allowed_entity_types=self.extractor_config.allowed_entity_types,
            allowed_relation_types=self.extractor_config.allowed_relation_types,
            allowed_relation_props=self.extractor_config.allowed_relation_props,
            allowed_entity_props=self.extractor_config.allowed_entity_props
        )

    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        self.graph_store.structured_query("MATCH (n) DETACH DELETE n")

    def process_papers_from_json(
        self,
        json_path: str,
        nrows: Optional[int] = None,
        clear_existing: bool = True
    ) -> PropertyGraphIndex:
        """
        Process papers from a JSON file and build the knowledge graph.
        
        Args:
            json_path: Path to the JSON file containing paper data
            nrows: Number of rows to process (None for all)
            clear_existing: Whether to clear existing graph data
            
        Returns:
            PropertyGraphIndex: The constructed knowledge graph index
        """
        # Clear existing graph if requested
        if clear_existing:
            self.clear_graph()

        # Read papers from JSON
        papers = pd.read_json(json_path, lines=True, nrows=nrows)
        
        # Convert papers to documents
        documents = [
            Document(text=f"{row['title']}: {row['abstract']}")
            for _, row in papers.iterrows()
        ]
        for document in documents:
            print(
                "The LLM sees this: \n",
                document.get_content(metadata_mode=MetadataMode.LLM),
                "Relationships: ",
                document.relationships
            )

        # Build and return the knowledge graph
        return self.build_graph(documents)

    def process_papers_from_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = 'title',
        abstract_col: str = 'abstract',
        clear_existing: bool = True
    ) -> PropertyGraphIndex:
        """
        Process papers from a pandas DataFrame and build the knowledge graph.
        
        Args:
            df: DataFrame containing paper data
            title_col: Name of the title column
            abstract_col: Name of the abstract column
            clear_existing: Whether to clear existing graph data
            
        Returns:
            PropertyGraphIndex: The constructed knowledge graph index
        """
        if clear_existing:
            self.clear_graph()

        documents = [
            Document(text=f"{row[title_col]}: {row[abstract_col]}")
            for _, row in df.iterrows()
        ]
        
        return self.build_graph(documents)

    def build_graph(self, documents: List[Document]) -> PropertyGraphIndex:
        """
        Build the knowledge graph from a list of documents.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            PropertyGraphIndex: The constructed knowledge graph index
        """
        return PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
            embed_kg_nodes=True,
            kg_extractors=[self.kg_extractor],
            show_progress=True
        )

    def query_graph(self, query: str) -> pd.DataFrame:
        """
        Execute a Cypher query against the Neo4j graph.
        
        Args:
            query: Cypher query string
            
        Returns:
            pd.DataFrame: Query results as a pandas DataFrame
        """
        result = self.graph_store.structured_query(query)
        return pd.DataFrame(result)

# Example usage
if __name__ == "__main__":

    neo4j_config = Neo4jConfig(
        username="neo4j",
        password="password",
        database="arxivCs"
    )
    
    extractor_config = ExtractorConfig(
        max_triplets_per_chunk=20,
        num_workers=4
    )
    
    kg_builder = LlamaIndexKGBuilder(
        neo4j_config=neo4j_config,
        llm_model="mistral",
        extractor_config=extractor_config
    )
    
    # Process papers
    graph_index = kg_builder.process_papers_from_json(
        "datasets/arxiv_cs_metadata.json",
        nrows=10,
        clear_existing=True
    )
    
    # Example query: ind first 10 entities
    entities = kg_builder.query_graph("""
        MATCH (n)
        RETURN n.text as entity, labels(n) as types
        LIMIT 10
    """)
    
    print("Sample entities:")
    print(entities)