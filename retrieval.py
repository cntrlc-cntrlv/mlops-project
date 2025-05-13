from langchain_qdrant.vectorstores import Qdrant
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from qdrant_client import QdrantClient
import os

# Set local model path
local_model_path = "C:/techai/program/model/bge-base-en-v1.5"  # Adjust the path as needed

# Check if model exists, otherwise download
if not os.path.exists(local_model_path):
    model_name = "BAAI/bge-base-en-v1.5"
else:
    model_name = local_model_path

# Load Embeddings Once
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Qdrant Connection
Qdrant_url = "https://85eb3094-2de3-497a-9fe6-5be30dbbb5be.us-east4-0.gcp.cloud.qdrant.io:6333"
Collection_name = "ssc"

client = QdrantClient(url=Qdrant_url, api_key=Qdrant_APIkey, prefer_grpc=False)

# Initialize Qdrant vector store
db = Qdrant(client=client, embeddings=embeddings, collection_name=Collection_name)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Function to Retrieve Documents
def get_relevant_docs(query):
    return retriever

print("Retriever initialized successfully!")
