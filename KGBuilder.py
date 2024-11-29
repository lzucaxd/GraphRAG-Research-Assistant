import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

json_path =  "datasets/arxiv_cs_metadata.json"
nrows = 5

papers = pd.read_json(json_path, lines=True, nrows=nrows)

documents = [
    Document(text=f"{row['title']}: {row['abstract']}")
    for i, row in papers.iterrows()
]


