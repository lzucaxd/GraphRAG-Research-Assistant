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
                
                # Create constraints for Authors
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (a:Author)
                    REQUIRE a.author_id IS UNIQUE
                """)
                
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
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id,
                    token=self.hf_token,
                    use_fast=False,  # Changed to False for better compatibility
                    padding_side='left'
                )
                
                logger.info("Tokenizer loaded successfully")
                
                # Set pad token if needed
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info("Set pad token to eos token")
                
                logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
                logger.info(f"Pad token: {tokenizer.pad_token}")

            except Exception as e:
                logger.error(f"Tokenizer initialization error: {str(e)}")
                raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")
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
        """Extract entities and relationships from an abstract with domain-agnostic prompt."""
        prompt = f"""[INST] Extract the key scientific entities and their relationships from this abstract. Include:
    - Main concepts, theories, or principles
    - Methods, techniques, or approaches
    - Objects of study or materials
    - Observed phenomena or results
    - Tools, measurements, or metrics used

    Rules:
    1. Respond with ONLY valid JSON - no other text or explanations
    2. Entity types should be general categories like "Concept", "Method", "Theory", "Material", "Phenomenon", "Metric"
    3. Use clear relationship types like "uses", "affects", "measures", "describes", "derives_from", "part_of"
    4. Entity names should preserve any technical notation or symbols
    5. Each entity must have a unique ID starting with "e" followed by a number
    6. Every relationship must connect existing entity IDs

    Abstract: "{abstract}"

    Expected JSON format:
    {{
        "entities": [
            {{
                "id": "e1",
                "name": "Riemann Hypothesis",
                "type": "Theory"
            }},
            {{
                "id": "e2",
                "name": "zeta function zeros",
                "type": "Concept"
            }},
            {{
                "id": "e3",
                "name": "quantum energy levels",
                "type": "Phenomenon"
            }}
        ],
        "relationships": [
            {{
                "source": "e1",
                "target": "e2",
                "type": "describes"
            }},
            {{
                "source": "e2",
                "target": "e3",
                "type": "analogous_to"
            }}
        ]
    }} [/INST]"""

        try:
            # Get model response with deterministic settings
            response = self.pipe(
                prompt,
                max_new_tokens=800,
                do_sample=False,
                top_p=1.0,
                repetition_penalty=1.15,
                pad_token_id=self.pipe.tokenizer.pad_token_id
            )[0]['generated_text']
            
            logger.info("Raw model response:")
            logger.info(response)
            
            
            def clean_json_text(text: str) -> str:
                """Clean and fix JSON string from model response with enhanced handling."""
            #     import re

            #     # Handle instruction markers and extract JSON portion
                parts = text.split('[/INST]')
                text = parts[-1].strip() if len(parts) > 1 else text
                
            #     # Find the outermost JSON structure
            #     json_start = text.find('{')
            #     json_end = text.rfind('}') + 1
            #     if json_start == -1 or json_end == 0:
            #         logger.warning("No JSON structure found in response")
            #         return "{}"
                
            #     text = text[json_start:json_end]

            #     # Basic cleanups
            #     text = text.replace("'", '"')  # Replace single quotes
            #     text = ''.join(char for char in text if ord(char) >= 32)  # Remove non-printable chars

            #     # Fix common JSON issues
            #     def fix_quotes(match):
            #         # Handle nested quotes in string values
            #         inner_text = match.group(1)
            #         # Escape any unescaped double quotes
            #         inner_text = inner_text.replace('"', '\\"')
            #         return f'"{inner_text}"'

            #     # Fix quoted values with nested quotes
            #     text = re.sub(r'"((?:[^"\\]|\\.)*)"(?=,|\s*})', fix_quotes, text)
                
            #     # Fix unquoted property names
            #     text = re.sub(r'(\w+):', r'"\1":', text)
                
            #     # Remove any trailing commas before closing brackets
            #     text = re.sub(r',(\s*[}\]])', r'\1', text)
                
            #     return text
                return text
            
            cleaned_json = clean_json_text(response)
            logger.info("Cleaned JSON text:")
            logger.info(cleaned_json)

            try:
                parsed_data = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Problem JSON: {cleaned_json}")
                logger.error(f"Error position: character {e.pos}")
                logger.error(f"Line {e.lineno}, column {e.colno}")
                
                # Advanced recovery attempts
                try:
                    # Try with all whitespace removed
                    stripped_json = re.sub(r'\s', '', cleaned_json)
                    parsed_data = json.loads(stripped_json)
                    logger.info("Succeeded parsing with stripped whitespace")
                except:
                    try:
                        # Try with relaxed JSON parsing using ast.literal_eval
                        import ast
                        parsed_data = ast.literal_eval(cleaned_json)
                        logger.info("Succeeded parsing with ast.literal_eval")
                    except:
                        logger.error("All parsing attempts failed")
                        return {"entities": [], "relationships": []}

            # Validate and normalize structure
            if not isinstance(parsed_data, dict):
                logger.warning("Parsed data is not a dictionary")
                return {"entities": [], "relationships": []}

            parsed_data.setdefault('entities', [])
            parsed_data.setdefault('relationships', [])

            # Enhanced entity validation with type normalization
            valid_entities = []
            entity_ids = set()
            for entity in parsed_data['entities']:
                if isinstance(entity, dict):
                    try:
                        # Normalize entity type
                        entity_type = str(entity.get('type', 'unknown')).strip()
                        if not entity_type:
                            entity_type = 'unknown'
                        
                        # Generate sequential ID if missing
                        entity_id = str(entity.get('id', f'e{len(entity_ids)+1}'))
                        
                        valid_entity = {
                            'id': entity_id,
                            'name': str(entity.get('name', 'unnamed')).strip(),
                            'type': entity_type
                        }
                        valid_entities.append(valid_entity)
                        entity_ids.add(entity_id)
                    except Exception as e:
                        logger.warning(f"Error processing entity: {entity}, Error: {str(e)}")
                        continue

            # Enhanced relationship validation with type normalization
            valid_relationships = []
            for rel in parsed_data['relationships']:
                if isinstance(rel, dict):
                    try:
                        source = str(rel.get('source', '')).strip()
                        target = str(rel.get('target', '')).strip()
                        if source in entity_ids and target in entity_ids:
                            rel_type = str(rel.get('type', 'related_to')).strip()
                            if not rel_type:
                                rel_type = 'related_to'
                            
                            valid_relationships.append({
                                'source': source,
                                'target': target,
                                'type': rel_type
                            })
                    except Exception as e:
                        logger.warning(f"Error processing relationship: {rel}, Error: {str(e)}")
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
            
            # Apply limit if specified
            if limit:
                df = df.limit(limit)
            
            # Include authors_parsed column
            df = df.select(['id', 'title', 'abstract', 'authors_parsed'])
            logger.info(f"Processing {len(df)} papers")
            
            # Process in batches
            for i in tqdm(range(0, len(df), self.config.batch_size)):
                batch = df.slice(i, self.config.batch_size)
                papers = batch.to_dicts()
                
                # # Preprocess authors and generate author nodes
                # for paper in papers:
                #     authors_parsed = paper.get('authors_parsed', [])
                #     authors = []
                #     for author in authors_parsed:
                #         last_name = author[0]
                #         first_name = author[1]
                #         # Create full name
                #         full_name = f"{first_name} {last_name}".strip()
                #         # Generate a unique author ID
                #         author_id = f"{last_name}_{first_name}".replace(' ', '_')
                #         authors.append({
                #             'author_id': author_id,
                #             'name': full_name
                #         })
                #     paper['authors'] = authors  # Replace authors_parsed with processed authors

                # Create paper and author nodes
                with self.driver.session() as session:
                    session.run("""
                        UNWIND $papers as paper
                        // Create Paper node
                        MERGE (p:Paper {id: paper.id})
                        SET p.title = paper.title,
                            p.abstract = paper.abstract
                        
                        // Create Author nodes and relationships from parsed authors
                        WITH p, paper
                        UNWIND paper.authors_parsed as author_data
                        MERGE (a:Author { 
                                author_id: reduce(s = author_data[0], x IN tail(author_data) | s + '_' + x) 
                        })
                        SET a.last_name = author_data[0],
                            a.first_name = author_data[1],
                            a.full_name = CASE 
                                WHEN author_data[1] = '' 
                                THEN author_data[0]
                                ELSE author_data[1] + ' ' + author_data[0]
                            END
                        MERGE (a)-[:AUTHORED]->(p)
                        """, papers=papers)

                # Process abstracts and create graph
                for paper in papers:
                    extracted = self.process_abstract(paper['abstract'])
                    if extracted['entities'] or extracted.get('relationships'):
                        self.create_graph_nodes(paper['id'], extracted)
                        
            logger.info("Paper processing complete")
                    
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