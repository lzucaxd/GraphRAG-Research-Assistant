import pandas as pd
from llama_index.core import Document
# Just runs .complete to make sure the LLM is listening
from llama_index.llms.ollama import Ollama
from typing import Any, List, Callable, Optional, Union, Dict
from IPython.display import Markdown, display

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)

from llama_index.core import PropertyGraphIndex

from llama_index.core import PropertyGraphIndex
import re

#Tweak the data set size with the nrows paramter
papers = pd.read_json("arxiv_cs.json", lines=True, nrows=10)

documents = [
    Document(text=f"{row['title']}: {row['abstract']}")
    for i, row in papers.iterrows()
]


#Replace your model here 
llm = Ollama(model="llama3.2:1b",request_timeout=300.0)
print(llm.model)


from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


import os


kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=20,
    num_workers=4,
    # Let the LLM infer entities and their labels (types) on the fly
    allowed_entity_types=None,
    # Let the LLM infer relationships on the fly
    allowed_relation_types=None,
    # LLM will generate any entity properties, set `None` to skip property generation (will be faster without)
    allowed_relation_props=[],
    # LLM will generate any relation properties, set `None` to skip property generation (will be faster without)
    allowed_entity_props=[],
)

dynamic_index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

dynamic_index.property_graph_store.save_networkx_graph(
    name="./KG_Papers.html"
)
