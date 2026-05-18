import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DOCUMENT_FOLDER = "./Doc"
VECTOR_DB_PATH = "./chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5