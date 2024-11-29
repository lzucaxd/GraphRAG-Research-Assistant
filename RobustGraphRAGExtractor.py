import backoff
import httpx
from typing import Optional, List, Tuple
import asyncio

import asyncio
import nest_asyncio

nest_asyncio.apply()

from typing import Any, List, Callable, Optional, Union, Dict
from IPython.display import Markdown, display
from GraphRAGExtractor import GraphRAGExtractor
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
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field


from llama_index.core.bridge.pydantic import Field
from typing import Any, List, Callable, Optional, Union, Dict

class RobustGraphRAGExtractor(GraphRAGExtractor):
    """Enhanced Graph Extractor with retry logic and better error handling.
    
    This class extends GraphRAGExtractor to add robust error handling and retry logic
    for dealing with timeout issues and other potential failures during extraction.
    """
    # Declare the new fields as part of the Pydantic model
    # max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    # base_delay: float = Field(default=1.0, description="Initial delay between retries in seconds")
    # max_delay: float = Field(default=60.0, description="Maximum delay between retries in seconds")
    
    # max_retries: int = Field(default=3)
    # base_delay: float = Field(default=1.0)
    # max_delay: float = Field(default=60.0)
    def __init__(
        self,
        llm: Optional[LLM] = Ollama(
            model="mistral",
            request_timeout=600.0,
            temperature=0.1
        ),
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        # max_retries: int = 3,
        # base_delay: float = 1,
        # max_delay: float = 60
    ) -> None:
        """Initialize the robust extractor.
        
        Args:
            llm: Language model to use for extraction
            extract_prompt: Prompt template for extraction
            parse_fn: Function to parse LLM output
            max_paths_per_chunk: Maximum number of paths to extract per chunk
            num_workers: Number of parallel workers
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
        """
        # Initialize the parent class first
        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
        )
        
        # # Initialize our new fields using the model's assignment
        # self.__dict__['max_retries'] = max_retries
        # self.__dict__['base_delay'] = base_delay
        # self.__dict__['max_delay'] = max_delay

    @backoff.on_exception(
        backoff.expo,
        (httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.TimeoutError),
        max_tries=3,
        base=1,
        max_value=60
    )
    async def _extract_with_retry(
        self, text: str, max_triplets: int
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Extract entities and relationships with retry logic."""
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=max_triplets,
            )
            return self.parse_fn(llm_response)
        except ValueError as e:
            print(f"Parsing error: {str(e)}")
            return [], []

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Enhanced extraction with better error handling and chunking."""
        assert hasattr(node, "text")
        text = node.get_content(metadata_mode="llm")

        # Initialize storage for extracted information
        all_entities = []
        all_relations = []

        try:
            # Extract entities and relationships with retry logic
            entities, relations = await self._extract_with_retry(
                text, self.max_paths_per_chunk
            )
            
            # Process successful extractions
            entity_metadata = node.metadata.copy()
            for entity, entity_type, description in entities:
                entity_metadata["entity_description"] = description
                entity_node = EntityNode(
                    name=entity,
                    label=entity_type,
                    properties=entity_metadata
                )
                all_entities.append(entity_node)

            relation_metadata = node.metadata.copy()
            for triple in relations:
                subj, obj, rel, description = triple
                relation_metadata["relationship_description"] = description
                rel_node = Relation(
                    label=rel,
                    source_id=subj,
                    target_id=obj,
                    properties=relation_metadata,
                )
                all_relations.append(rel_node)

        except Exception as e:
            print(f"Extraction error for node: {str(e)}")
            # Continue with any successfully extracted information

        # Update node metadata
        node.metadata[KG_NODES_KEY] = all_entities
        node.metadata[KG_RELATIONS_KEY] = all_relations
        return node