import argparse
from GraphRAG import GraphRAG

"""
Takes in command line arguments to create the GraphRAG object.
Facilitates interactive querying of the GraphRAG object.
"""
def main():

    parser = argparse.ArgumentParser(description='Process research papers data')
    parser.add_argument('-j', '--json_path', type=str, default="datasets/arxiv_cs_metadata.json",
                      help='Path to JSON file containing papers data')
    parser.add_argument('-n', '--nrows', type=int, default=5,
                      help='Number of rows to read from JSON')
    parser.add_argument('-d', '--database', type=str, default="neo4j",
                      help='Name of the database')
    parser.add_argument('-l', '--llm', type=str, default="qwen2.5",
                      help='Ollama Model name')
    parser.add_argument('-e', '--embed-model', type=str, default="avsolatorio/GIST-all-MiniLM-L6-v2",
                      help='HuggingFace Embedding Model Name')
    
    args = parser.parse_args()
    
    graph_rag = GraphRAG(args.json_path, args.nrows, args.database, args.llm, args.embed_model)

    while True:
        query_str = input("Enter a query, or 'exit' to quit: ")
        if query_str.lower() == "exit":
            print("Goodbye!")
            break
        graph_rag.query(query_str)
    
if __name__ == "__main__":
    main()