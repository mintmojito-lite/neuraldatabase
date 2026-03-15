import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os

def generate():
    print("Loading ag_news dataset...")
    dataset = load_dataset("ag_news", split="train")
    
    # Take 100k for index, next 1k for queries
    all_texts = dataset["text"]
    train_texts = all_texts[:100000]
    query_texts = all_texts[100000:101000]
    
    print(f"Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding 100k sentences (this may take a few minutes)...")
    embeddings_100k = model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings_100k = embeddings_100k.astype(np.float32)
    
    print("Encoding 1k queries...")
    queries_1k = model.encode(query_texts, show_progress_bar=True, convert_to_numpy=True)
    queries_1k = queries_1k.astype(np.float32)
    
    # Save files
    print("Saving files...")
    np.save("embeddings_100k.npy", embeddings_100k)
    np.save("queries_1k.npy", queries_1k)
    
    with open("sentences_100k.txt", "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(text.replace("\n", " ") + "\n")
            
    print("Done!")
    print(f"embeddings_100k.npy: {embeddings_100k.shape}")
    print(f"queries_1k.npy: {queries_1k.shape}")

if __name__ == "__main__":
    generate()
