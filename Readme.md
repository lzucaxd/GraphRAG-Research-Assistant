1. Setting Up Ollama Locally
    Follow the instructions provided here to install Ollama on your local machine.
    Ensure that Ollama is running locally on a terminal usung below command before executing any Python files:
    bash

    ```ollama run mistral```

2. Neo4j Setup with Docker
    This project uses Neo4j hosted in a Docker container on a virtual machine. To replicate this setup:

    Install Docker Desktop by following the installation guide here.

    Run the following command to start Neo4j:



```
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=["apoc"] \
    neo4j:latest
```