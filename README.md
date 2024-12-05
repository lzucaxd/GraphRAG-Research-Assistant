## Setup instructions:

### Platform/Specs:
- This project was developed on, and has only been tested on macOS. Try other platforms at your own risk.
- An M1 Max with 32GB of unified memory was sufficient to run our local LLM: qwen2.5 (7b params default in Ollama).
- Other local models were more prone to hallucination, and caused some parts of the pipeline to break due to inability to adhere to directions. 

### Prerequisites:
- Install the desktop version of Ollama: https://ollama.com/download
- Install the desktop version of Neo4j: https://neo4j.com/download/

### Before you run:
- Create your conda environment (Python 3.11.10 was used for this project)
- Create a Project in Neo4j![alt text](images/start_neo4j.png)
- Install the APOC plugin to your DBMS.![alt text](images/install_plugin.gif) 
- Start your Neo4j DBMS (You'll need to set a password the first time). 
  - This should be accessible on your browser at `bolt://localhost:7687`.
  - I keep the default username "neo4j', and use "password" as the password. 
  - Each project can have multiple databases. I highly suggest using the default db, also called "neo4j"
- Run `pip install -r requirements.txt`
- In your terminal run: `ollama run qwen2.5`. You can change the model but we do not guarantee results with other models.
- While our methods for cleaning/creating `datasets/arxiv_cs_metadata.json` are included in `src/PullDataset.py`, we reccomend using the default, included, data set for best results/consistency. 

### How to run?
You have 2 options for running our project:
> WARNING: This will take some time. Streamlit is reccomended because it caches some of the setup processes.
1. CLI
	- run `python src/main.py`. If you used our suggested default values you won't need to specify any of these command line arguments:
    	- ` -j <dataset_json_path>, -n <nrows_from_dataset> -d <neo4j_db_name> -l <ollama_llm_model_name> -e <HuggingFace_embedding_model_name> `
2. Streamlit UI
   - run `streamlit run src/main_gui.py`