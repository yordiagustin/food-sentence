import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

PROCESSED_DATA_FILE = 'processed_recipes.csv'
EMBEDDINGS_FILE = 'recipe_embeddings.npy'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def generate_and_save_embeddings(data_file, output_file):
    print("Starting the embedding generation process...")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found in '{data_file}'.")
        print("Please run the preprocessing script first.")
        return

    df = pd.read_csv(data_file)
    
    df['recipe_name'] = df['recipe_name'].fillna('')
    df['ingredients_text'] = df['ingredients_text'].fillna('')
    df['instructions'] = df['instructions'].fillna('')

    df['embedding_text'] = (
        df['recipe_name'] + ". " +
        "Ingredientes: " + df['ingredients_text'] + ". " +
        "Preparación: " + df['instructions']
    )

    print(f"Generating embeddings for {len(df)} recipes.")
    print("Example text for embeddings:")
    print(f"'{df['embedding_text'].iloc[0][:200]}...'")

    print(f"Loading model '{MODEL_NAME}'... (This may take a moment the first time)")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df['embedding_text'].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32  # Batch size for better performance
    )

    np.save(output_file, embeddings)

    print("-" * 60)
    print(f"Success! Embeddings generated and saved in '{output_file}'")
    print(f"Shape of the embeddings array: {embeddings.shape}")
    print("-" * 60)
    
    if len(df) != len(embeddings):
        print(f"WARNING: Mismatch in dimensions!")
        print(f"Recipes: {len(df)}, Embeddings: {len(embeddings)}")
    else:
        print("✓ The dimensions of recipes and embeddings match correctly")

if __name__ == "__main__":
    generate_and_save_embeddings(PROCESSED_DATA_FILE, EMBEDDINGS_FILE)