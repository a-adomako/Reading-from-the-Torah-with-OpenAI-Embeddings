from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']
word_to_compare_1 = input("Enter a word: ")
word_to_compare_2 = input("Enter a second word: ")

def main():
    # Get embedding for the word
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query(f"{word_to_compare_1}")
    """
    print(f"Vector for {word_to_compare_1}: {vector}")
    print(f"Vector length: {len(vector)}")
    """
    # Compare the vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = (f"{word_to_compare_1}", f"{word_to_compare_2}")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")
    

if __name__ == "__main__":
        main()