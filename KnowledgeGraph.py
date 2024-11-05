import pandas as pd
from llama_index.core import Document, PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
)
from llama_index.core import StorageContext

#Tweak the data set size with the nrows paramter
papers = pd.read_json("datasets/arxiv_cs_metadata.json", lines=True, nrows=10)

documents = [
    Document(text=f"{row['title']}: {row['abstract']}")
    for i, row in papers.iterrows()
]

llm = Ollama(model="mistral", request_timeout=300.0)
print(llm.model)

graph_store = Neo4jPropertyGraphStore(
        username="neo4j",
        password="password",
        url="bolt://localhost:7687",
        database="arxivCs",
)

storage_context = StorageContext.from_defaults(
    property_graph_store=graph_store
)

graph_store.query("MATCH (n) DETACH DELETE n")

kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=20,
    num_workers=4,
    allowed_entity_types=None,
    allowed_relation_types=None,
    allowed_relation_props=[],
    allowed_entity_props=[],
)

dynamic_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)


