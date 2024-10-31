from dataclasses import dataclass
from neo4j import GraphDatabase
import polars as pl
from tqdm import tqdm
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kg_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    batch_size: int = 4
    max_length: int = 2048
    temperature: float = 0.7

class MistralKGBuilder:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Check for HF token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
            
        # Setup Neo4j
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD not found in .env file")
            
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password
            )
        )
        
        # Initialize database first
        self.init_database()
        
        # Setup model
        self.config = Config()
        self.setup_model()
    
    def init_database(self):
        """Setup Neo4j constraints"""
        try:
            with self.driver.session() as session:
                # Create constraints for Papers
                session.run("""
                    CREATE CONSTRAINT paper_id IF NOT EXISTS 
                    FOR (p:Paper) REQUIRE p.id IS UNIQUE
                """)
                
                # Create constraints for Entities
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise RuntimeError(f"Failed to initialize database: {str(e)}")
    
    def setup_model(self):
        """Initialize Mistral with M1 optimizations"""
        try:
            logger.info("Setting up model...")
            
            # Check device
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon) device")
            else:
                device = "cpu"
                logger.info("Using CPU device")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                token=self.hf_token,
                trust_remote_code=True,
                use_fast=False  # Use slow tokenizer for compatibility

            )
            
            logger.info("Tokenizer loaded successfully")
            
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
            logger.info(f"Pad token: {tokenizer.pad_token}")
            # Load model with M1 optimizations
            logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                token=self.hf_token,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map='auto',
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
            # Create pipeline
            logger.info("Creating pipeline...")
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # device=device,
                torch_dtype=torch.float16,
                framework="pt"
            )
            
            logger.info("Pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Error during model setup: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def create_graph_nodes(self, paper_id: str, data: dict):
        """Create nodes and relationships in Neo4j"""
        try:
            with self.driver.session() as session:
                # Create entities and their relationships to papers
                if data['entities']:
                    session.run("""
                        MATCH (p:Paper {id: $paper_id})
                        UNWIND $entities as entity
                        MERGE (e:Entity {id: entity.id})
                        SET e.name = entity.name,
                            e.type = entity.type
                        MERGE (p)-[:MENTIONS]->(e)
                        """, paper_id=paper_id, entities=data['entities'])
                
                # Create relationships between entities
                if data.get('relationships'):
                    session.run("""
                        UNWIND $relationships as rel
                        MATCH (e1:Entity {id: rel.source})
                        MATCH (e2:Entity {id: rel.target})
                        MERGE (e1)-[r:RELATES_TO {type: rel.type}]->(e2)
                        """, relationships=data['relationships'])
                        
                logger.debug(f"Created graph nodes for paper {paper_id}")
                
        except Exception as e:
            logger.error(f"Error creating graph nodes for paper {paper_id}: {str(e)}")
            raise
    
    def process_abstract(self, abstract: str) -> dict:
        """Extract entities and relationships using Mistral with enhanced debug logging"""
        prompt = f"""<s>[INST] Extract key entities and their relationships from this scientific abstract.
        You must respond with ONLY valid JSON, properly formatted with double quotes.
        
        Abstract: {abstract}
        
        Format your response EXACTLY like this example, with no other text:
        {{
            "entities": [
                {{
                    "id": "e1",
                    "name": "machine learning",
                    "type": "technology"
                }}
            ],
            "relationships": [
                {{
                    "source": "e1",
                    "target": "e2",
                    "type": "uses"
                }}
            ]
        }} [/INST]</s>"""
        
        try:
            # Clear GPU memory before inference
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Get model response
            response = self.pipe(
                prompt,
                max_new_tokens=500,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=self.pipe.tokenizer.pad_token_id
            )[0]['generated_text']
            
            logger.debug("Raw model response:")
            logger.debug(response)
            
            # Extract JSON portion
            json_text = response.split('[/INST]')[-1].strip()
            logger.debug("Extracted JSON text:")
            logger.debug(json_text)
            
            def clean_json_text(text):
                # Find JSON boundaries
                start = text.find('{')
                end = text.rfind('}') + 1
                
                if start == -1 or end == 0:
                    logger.warning("No JSON structure found in response")
                    return "{}"
                
                # Extract JSON portion
                text = text[start:end]
                logger.debug("JSON after boundary extraction:")
                logger.debug(text)
                
                # Remove any leading/trailing whitespace
                text = text.strip()
                
                # Replace single quotes with double quotes
                text = text.replace("'", '"')
                
                # Fix property names
                text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
                
                # Remove trailing commas
                text = re.sub(r',(\s*[}\]])', r'\1', text)
                
                # Remove newlines and extra spaces
                text = re.sub(r'\s+', ' ', text)
                
                # Fix common formatting issues
                text = text.replace('"{', '{').replace('}"', '}')
                text = text.replace('[]', '""')
                
                logger.debug("Cleaned JSON:")
                logger.debug(text)
                
                return text
            
            cleaned_json = clean_json_text(json_text)
            
            try:
                parsed_data = json.loads(cleaned_json)
                logger.debug("Successfully parsed JSON:")
                logger.debug(parsed_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Problem JSON: {cleaned_json}")
                logger.error(f"Error position: character {e.pos}")
                logger.error(f"Line {e.lineno}, column {e.colno}")
                
                # Try parsing with more lenient cleaning
                try:
                    # Remove all whitespace and try again
                    stripped_json = re.sub(r'\s', '', cleaned_json)
                    parsed_data = json.loads(stripped_json)
                    logger.info("Succeeded parsing with stripped whitespace")
                except:
                    logger.error("Failed even with stripped whitespace")
                    return {"entities": [], "relationships": []}
            
            # Validate structure
            if not isinstance(parsed_data, dict):
                logger.warning("Parsed data is not a dictionary")
                return {"entities": [], "relationships": []}
            
            # Ensure required keys exist
            parsed_data.setdefault('entities', [])
            parsed_data.setdefault('relationships', [])
            
            # Validate entities
            valid_entities = []
            entity_ids = set()
            for entity in parsed_data['entities']:
                if isinstance(entity, dict):
                    try:
                        valid_entity = {
                            'id': str(entity.get('id', f'e{len(entity_ids)+1}')),
                            'name': str(entity.get('name', 'unnamed')),
                            'type': str(entity.get('type', 'unknown'))
                        }
                        valid_entities.append(valid_entity)
                        entity_ids.add(valid_entity['id'])
                    except Exception as e:
                        logger.warning(f"Error processing entity: {entity}")
                        continue
            
            # Validate relationships
            valid_relationships = []
            for rel in parsed_data['relationships']:
                if isinstance(rel, dict):
                    try:
                        if rel.get('source') in entity_ids and rel.get('target') in entity_ids:
                            valid_relationships.append({
                                'source': str(rel['source']),
                                'target': str(rel['target']),
                                'type': str(rel.get('type', 'related_to'))
                            })
                    except Exception as e:
                        logger.warning(f"Error processing relationship: {rel}")
                        continue
            
            result = {
                "entities": valid_entities,
                "relationships": valid_relationships
            }
            
            logger.debug("Final validated output:")
            logger.debug(result)
            return result
                
        except Exception as e:
            logger.error(f"Error in process_abstract: {str(e)}")
            if 'response' in locals():
                logger.error(f"Raw response was: {response}")
            return {"entities": [], "relationships": []}
    
    def process_papers(self, input_path: str, limit: int = None):
        """Process papers and build knowledge graph"""
        try:
            # Read papers efficiently using Polars
            if input_path.endswith('.json'):
                df = pl.read_ndjson(input_path)
            elif input_path.endswith('.csv'):
                df = pl.scan_csv(input_path)
            elif input_path.endswith('.parquet'):
                df = pl.scan_parquet(input_path)
            else:
                raise ValueError("Unsupported file format. Use JSON, CSV, or Parquet.")
            
            # Apply limit if specified
            if limit:
                df = df.limit(limit)
            
            # Collect necessary columns
            df = df.select(['id', 'title', 'abstract'])
            logger.info(f"Processing {len(df)} papers")
            
            # Process in batches
            for i in tqdm(range(0, len(df), self.config.batch_size)):
                batch = df.slice(i, self.config.batch_size)
                
                # Create paper nodes
                with self.driver.session() as session:
                    session.run("""
                        UNWIND $papers as paper
                        MERGE (p:Paper {id: paper.id})
                        SET p.title = paper.title,
                            p.abstract = paper.abstract
                        """, papers=batch.to_dicts())
                
                # Process abstracts and create graph
                for paper in batch.to_dicts():
                    extracted = self.process_abstract(paper['abstract'])
                    if extracted['entities'] or extracted.get('relationships'):
                        self.create_graph_nodes(paper['id'], extracted)
                
                # Clear memory after each batch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing papers: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Initialize builder
        logger.info("Initializing MistralKGBuilder...")
        builder = MistralKGBuilder()
        
        # Process a small test batch
        logger.info("Starting paper processing...")
        builder.process_papers(
            "/Users/agastyadas/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/203/arxiv-metadata-oai-snapshot.json",
            limit=5  # Start with 5 papers as a test
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure your .env file contains a valid HUGGINGFACE_TOKEN")
        print("2. Try running 'huggingface-cli login' in terminal")
        print("3. Check your internet connection")
        print("4. Verify you have at least 16GB of free RAM")
        sys.exit(1)