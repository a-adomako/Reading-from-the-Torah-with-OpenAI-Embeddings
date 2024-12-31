from langchain_community.embeddings import OpenAIEmbeddings
import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    # Initialize embedding function
    embedding_function = OpenAIEmbeddings()

    # Input word and candidates
    word_to_compare = "apple"
    candidate_words = ["orange", "banana", "laptop", "technology"]

    # Get embedding for the target word
    vector = embedding_function.embed_query(word_to_compare)

    # Get embeddings for candidate words
    candidates_embeddings = {word: embedding_function.embed_query(word) for word in candidate_words}

    # Find the closest word
    closest_word = None
    highest_similarity = -1  # Initialize with a low similarity value

    for candidate, candidate_vector in candidates_embeddings.items():
        similarity = cosine_similarity(vector, candidate_vector)
        print(f"Similarity between {word_to_compare} and {candidate}: {similarity}")

        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_word = candidate

    print(f"The closest word to '{word_to_compare}' is '{closest_word}' with similarity {highest_similarity:.4f}")

if __name__ == "__main__":
    main()