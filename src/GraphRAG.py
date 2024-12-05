
import re
import time
import pandas as pd
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from GraphRAGExtractor import GraphRAGExtractor
from GraphRAGStore import GraphRAGStore
from GraphRAGQueryEngine import GraphRAGQueryEngine
from pyvis.network import Network

"""
This class initializes sets up the subcomponents required for the GraphRAG system, and controls
their interactions: 
- GraphRAGStore
- GraphRAGExtractor
- GraphRAGQueryEngine
"""
class GraphRAG():
 

    KG_TRIPLET_EXTRACT_TMPL = """
        -Goal-
        Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
        Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

        -Steps-
        1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_type: Type of the entity
        - entity_description: Comprehensive description of the entity's attributes and activities
        Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

        2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
        For each pair of related entities, extract the following information:
        - source_entity: name of the source entity, as identified in step 1
        - target_entity: name of the target entity, as identified in step 1
        - relation: relationship between source_entity and target_entity
        - relationship_description: explanation as to why you think the source entity and the target entity are related to each other

        Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

        3. When finished, output.

        -Real Data-
        ######################
        text: {text}
        ######################
        output:"""

        
    def __init__(self, json_path, nrows, database, llm, embed_model):
        print(f"Initializing GraphRAG...")
        self.json_path = json_path
        self.nrows = nrows
        self.database = database
        self.llm = Ollama(model=llm,  request_timeout=20000)
        self.embed_model = HuggingFaceEmbedding(embed_model)
        
        
        self.graph_store = GraphRAGStore(
            username="neo4j", 
            password="password", 
            url="bolt://localhost:7687", 
            database=self.database,
            llm=self.llm,
        )
        print(f"GraphRAGStore initialized with database: {database}")

        self.kg_extractor = GraphRAGExtractor(
            llm=self.llm,
            extract_prompt=self.KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=20,
            num_workers=4,
            parse_fn=self.parse_fn,
        )
        print(f"GraphRAGExtractor initialized.")
        nodes = self.create_nodes_from_json()
        self.index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[self.kg_extractor],
            property_graph_store=self.graph_store,
            llm=self.llm,
            show_progress=True,
            embed_model=self.embed_model,
        )
        try:
            print(f"Building communities...")
            self.index.property_graph_store.build_communities()
            print(f"Communities built.")
            self.save_community_graph()
            print(f"Community graph saved.")
        except Exception as e:
            print(f"Error building communities:")
            print(e)
        
        self.query_engine = GraphRAGQueryEngine(
            graph_store=self.index.property_graph_store,
            llm=self.llm,
            index=self.index,
            similarity_top_k=10,
        )
        print(f"GraphRAG initialized, and ready for queries.")

       
    def create_nodes_from_json(self):
        papers = pd.read_json(self.json_path, lines=True, nrows=self.nrows)
        print(f"Reading {self.nrows} rows from {self.json_path}")
        print(f"Found papers:")
        for i, row in papers.iterrows():
            print(f"{i+1}. {row['title']}")
        
        documents = [
            Document(text=f"{row['title']}: {row['abstract']}",)
            for i, row in papers.iterrows()
        ]

        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        )
    
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"{len(nodes)} nodes created from {len(documents)} documents.")
        return nodes

    def parse_fn(self, response_str: str):
        # Regex that helps us extract the entities and relationships from the LLM output
        entity_pattern = r'\("entity"\$\$\$\$(.*?)\$\$\$\$(.*?)\$\$\$\$(.*?)\)'
        relationship_pattern = r'\("relationship"\$\$\$\$(.*?)\$\$\$\$(.*?)\$\$\$\$(.*?)\$\$\$\$(.*?)\)'
        
        entities = re.findall(entity_pattern, response_str, re.DOTALL)
        relationships = re.findall(relationship_pattern, response_str, re.DOTALL)
        
        entities = [(e1.strip(), e2.strip(), e3.strip()) for e1, e2, e3 in entities]
        relationships = [(r1.strip(), r2.strip(), r3.strip(), r4.strip()) 
                        for r1, r2, r3, r4 in relationships]
        
        # Here I add default values if the entities or relationships are empty  
        # -- This tends to happen when our local LLM hallucinates, and doesn't follow the specified formatting
        if not entities:
            print("Using Dummy Entities")
            entities = [("DummyEntityName", "DummyEntityType", "DummyEntityDescription")]
        if not relationships:
            print("Using Dummy Relationships")
            relationships = [("DummyRelationshipSourceEntity", "DummyRelationshipTargetEntity", "DummyR", "DummyR")]
        
        print(f"Found entities: {entities}")
        print(f"Found relationships: {relationships}")
        
        return entities, relationships
    
    # This lets us visualize the community graph
    def save_community_graph(self):
        net = Network(
            directed = True,
            select_menu = True, 
            filter_menu = True, 
        )
        net.show_buttons() 
        net.from_nx(self.graph_store._create_nx_graph()) 
        net.write_html('community_graph.html')
    
    def query(self, query_str):
        print(f"Querying GraphRAG with: {query_str}")
        response = self.query_engine.query(query_str)
        print(f"Response: {response}")
        return response