import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize Chroma client (persistent)
client = chromadb.PersistentClient(path="./chromadb_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create collection
collection = client.get_or_create_collection(name="transcripts", embedding_function=ef)

# Load and chunk txt files
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Process files
for file in os.listdir("./transcripts"):
    if file.endswith('.txt'):
        with open(f"./transcripts/{file}", 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = chunk_text(content)
            
            # Add to collection
            collection.add(
                documents=chunks,
                metadatas=[{"source": file, "chunk_id": i} for i in range(len(chunks))],
                ids=[f"{file}_{i}" for i in range(len(chunks))]
            )

print(f"✅ Added {collection.count()} chunks from transcripts!")
