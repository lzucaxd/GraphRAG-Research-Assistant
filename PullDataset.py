import polars as pl
import os
import json
import logging
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = "/Users/agastyadas/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/203"
INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "datasets/arxiv_cs_metadata.json"
CS_CATEGORIES = ['cs.CV', 'cs.LG', 'cs.CL', 'cs.AI', 'cs.NE', 'cs.RO']

def clean_paper_data(row: dict) -> dict:
    """Clean and validate a single paper's data"""
    # Clean authors data
    authors_parsed = row.get('authors_parsed', [])
    if not isinstance(authors_parsed, list) or not authors_parsed:
        authors_parsed = [["Unknown", "", ""]]
    else:
        # Clean each author entry while preserving original structure
        cleaned_authors = []
        for author in authors_parsed:
            if isinstance(author, list):
                # Just clean the strings without modifying the structure
                author_entry = [str(name).strip() if name else "" for name in author]
                cleaned_authors.append(author_entry)
        authors_parsed = cleaned_authors if cleaned_authors else [["Unknown", "", ""]]

    #TODO handle other values
    return {
        'id': str(row.get('id', '')).strip(),
        'title': str(row.get('title', '')).strip(),
        'abstract': str(row.get('abstract', '')).strip(),
        'authors_parsed': authors_parsed
    }

# Create output directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)

try:
    # Read the dataset
    logger.info("Reading ArXiv dataset...")
    input_path = os.path.join(DATASET_PATH, INPUT_FILE)
    df = pl.read_ndjson(input_path)
    
    # Create category pattern for filtering
    pattern = r"\b(?:" + "|".join(CS_CATEGORIES).replace(".", r"\.") + r")\b"
    
    # Filter for CS categories
    logger.info("Filtering CS papers...")
    filtered_df = df.filter(pl.col("categories").str.contains(pattern, strict=True))
    
    # Select and clean needed columns
    logger.info("Processing papers...")
    processed_records = []
    
    # Process each row
    for row in filtered_df.iter_rows(named=True):
        try:
            cleaned_record = clean_paper_data(row)
            if all(cleaned_record.values()):  # Ensure no empty values
                processed_records.append(cleaned_record)
        except Exception as e:
            logger.warning(f"Error processing paper {row.get('id', 'unknown')}: {str(e)}")
            continue
    
    # Write the cleaned records to file
    logger.info(f"Writing {len(processed_records)} papers to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in processed_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info("Successfully completed preprocessing!")
    logger.info(f"Processed papers: {len(processed_records)}")

except Exception as e:
    logger.error(f"Error during preprocessing: {str(e)}")
    raise

# Verify the output
try:
    # Try reading back the processed file to verify
    test_df = pl.read_ndjson(OUTPUT_FILE)
    logger.info(f"Verification successful. Output file contains {len(test_df)} papers.")
    
    # Show sample of the data
    if len(test_df) > 0:
        sample = test_df.head(1).to_dicts()[0]
        logger.info("Sample paper structure:")
        logger.info(json.dumps(sample, indent=2))

except Exception as e:
    logger.error(f"Error verifying output file: {str(e)}")