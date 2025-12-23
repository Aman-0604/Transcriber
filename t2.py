import requests
import chromadb
from chromadb.utils import embedding_functions

class ChromaRAGChat:
    def __init__(self, chroma_path="./chromadb_db"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self.collection = self.client.get_collection("transcripts")
        self.server_url = "http://localhost:8080/v1/chat/completions"
    
    def retrieve(self, query, k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return "\n\n".join(results['documents'][0])
    
    def chat(self, message, history=""):
        # Retrieve relevant chunks
        context = self.retrieve(message)
        
        # Build prompt
        prompt = f"""Use only the following transcript context to answer. If unrelated, say "No relevant information found."

CONTEXT:
{context}

CHAT HISTORY:
{history}

QUESTION: {message}

ANSWER:"""
        
        # Hit llama server
        payload = {
            "model": "llama-3.2-1b",  # your model name
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        response = requests.post(self.server_url, json=payload)
        answer = response.json()["choices"][0]["message"]["content"]
        
        return answer

# Run chat
chatbot = ChromaRAGChat()
history = ""

print("Chat ready! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit': break
    
    answer = chatbot.chat(user_input, history)
    print("AI:", answer)
    
    history += f"\nQ: {user_input}\nA: {answer[:150]}..."  # Truncate
