import polars as pl
import kagglehub

# class DatasetLoader:
#     def __init__(self, dataset_name: str):
#         self.dataset_name = dataset_name
#         self.dataset_path = self.download_dataset()

#     def download_dataset(self):
#         """Download the dataset from Kaggle"""
#         path = kagglehub.dataset_download(self.dataset_name)
#         print("Path to dataset files:", path)
#         return path

#     def load_data(self, file_name: str) -> pl.DataFrame:
#         """Load data from a specified file"""
#         return pl.read_ndjson(file_name)

#     def filter_data(self, df: pl.DataFrame, category_pattern: str) -> pl.DataFrame:
#         """Filter data based on category pattern"""
#         return df.filter(pl.col("categories").str.contains(category_pattern, strict=True))
# # Download latest version
# path = kagglehub.dataset_download("Cornell-University/arxiv")

# print("Path to dataset files:", path)

DATASET_PATH = "/Users/agastyadas/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/203"
# Loading the entire JSON dataset from the input file path
cs_arxiv_df = pl.read_ndjson(DATASET_PATH + "/arxiv-metadata-oai-snapshot.json")


# Filtering rows where the 'categories' column contains specific computer science categories
cs_arxiv_df_filtered = cs_arxiv_df.filter(pl.col("categories").str.contains(r"\b(?:cs\.(?:CV|LG|CL|AI|NE|RO))\b", strict=True))
print(cs_arxiv_df_filtered['abstract'][0])

