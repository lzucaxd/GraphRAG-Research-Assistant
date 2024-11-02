1. Install neo4j desktop 
2. pip install -r requirements.txt
3. Create a huggingface token 
4. set up your .env with hf token and db information
5. Before first run, use kaggle to download dataset
6. Set your dataset path in PullDataset.py
7. Run pull dataset.py
8. Now the filtered arxiv data should be in datasets/arxiv_cs_metadata.json
9. Run knowledgeGraph.py
