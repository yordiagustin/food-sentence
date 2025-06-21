import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uvicorn

PROCESSED_DATA_FILE = 'processed_recipes.csv'
EMBEDDINGS_FILE = 'recipe_embeddings.npy'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

app = FastAPI(
    title="Recipe Recommender API",
    description="API for recipe recommendations using semantic embeddings",
    version="1.0.0"
)

df = None
recipe_embeddings = None
model = None

@app.on_event("startup")
async def load_resources():
    """Load resources at startup"""
    global df, recipe_embeddings, model
    
    try:
        print("Loading resources...")
        
        if not os.path.exists(PROCESSED_DATA_FILE):
            raise FileNotFoundError(f"Not found: {PROCESSED_DATA_FILE}")
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Not found: {EMBEDDINGS_FILE}")
        
        df = pd.read_csv(PROCESSED_DATA_FILE)
        recipe_embeddings = np.load(EMBEDDINGS_FILE)
        model = SentenceTransformer(MODEL_NAME)
        
        if len(df) != len(recipe_embeddings):
            raise ValueError(f"Mismatch: {len(df)} recetas vs {len(recipe_embeddings)} embeddings")
        
        print(f"API lista: {len(df)} recetas cargadas")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

class RecipeRequest(BaseModel):
    user_query: str = Field(
        ...,
        example="creamy soup without dairy",
        description="Lo que el usuario quiere cocinar, incluyendo restricciones"
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de recomendaciones"
    )

class RecipeResponse(BaseModel):
    recipe_name: str
    similarity_score: float
    ingredients_text: Optional[str] = None
    instructions: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None

@app.post("/recommend", response_model=List[RecipeResponse])
async def recommend_recipes(request: RecipeRequest):
    """Main Endpoint for recipe recommendations using semantic embeddings"""
    try:

        if df is None or recipe_embeddings is None or model is None:
            raise HTTPException(status_code=500, detail="Resources not loaded")
        
        query_embedding = model.encode([request.user_query], convert_to_numpy=True)
        
        similarities = cosine_similarity(query_embedding, recipe_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:request.top_n]
        
        recommendations = []
        for idx in top_indices:
            recipe = df.iloc[idx]
            
            recommendation = RecipeResponse(
                recipe_name=str(recipe.get('recipe_name', 'No name')),
                similarity_score=float(similarities[idx]),
                ingredients_text=str(recipe.get('ingredients_text', 'Not available')),
                instructions=str(recipe.get('instructions', 'Not available')),
                source=str(recipe.get('source', 'Unknown')),
                language=str(recipe.get('language', 'unknown'))
            )
            
            recommendations.append(recommendation)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)