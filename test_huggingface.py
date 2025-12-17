from sentence_transformers import SentenceTransformer

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    texts = [
        "Python developer with 2 years experience",
        "Java backend engineer",
        "Data scientist skilled in Python and ML"
    ]

    embeddings = model.encode(texts)

    print("Embedding shape:", embeddings.shape)
    print("First embedding (first 10 values):")
    print(embeddings[0][:10])

if __name__ == "__main__":
    main()