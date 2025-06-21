import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time
import logging
import uvicorn

PROCESSED_DATA_FILE = 'processed_recipes.csv'
EMBEDDINGS_FILE = 'recipe_embeddings.npy'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safe Recipe Recommender API",
    description="Safe recipe recommender system for people with allergies",
    version="1.1.0"
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
        
        print(f"API ready: {len(df)} recipes loaded")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

class RecipeRequest(BaseModel):
    user_query: str = Field(
        ...,
        example="algo cremoso para cenar",
        description="Descripción de lo que quieres cocinar"
    )
    allergies: List[str] = Field(
        ...,
        example=["Dairy", "Nuts"],
        description="Lista de alergias alimentarias a evitar (OBLIGATORIO para seguridad)"
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

class InferenceMetrics(BaseModel):
    total_time_ms: float
    embedding_time_ms: float
    similarity_time_ms: float
    processing_time_ms: float
    query_length: int
    total_recipes_in_db: int
    safe_recipes_after_filtering: int
    recipes_returned: int
    top_n: int
    avg_similarity_score: float
    max_similarity_score: float
    min_similarity_score: float
    allergies_filtered: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[RecipeResponse]
    metrics: InferenceMetrics

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_safe_recipes(request: RecipeRequest):
    """
    Recommends safe recipes for people with allergies
    """
    start_time = time.time()
    
    try:
        if df is None or recipe_embeddings is None or model is None:
            raise HTTPException(status_code=500, detail="Resources not loaded")
        
        logger.info(f"SAFE RECIPE REQUEST")
        logger.info(f"   • Query: '{request.user_query}'")
        logger.info(f"   • Allergies: {request.allergies}")
        logger.info(f"   • Top N: {request.top_n}")
        
        logger.info(f"Safety filtering...")
        safe_df = df.copy()
        safe_indices = list(range(len(df)))
        
        if request.allergies:
            for allergy in request.allergies:
                column_name = f'contains_{allergy}'
                if column_name in safe_df.columns:

                    initial_count = len(safe_df)
                    mask = safe_df[column_name] == False
                    safe_df = safe_df[mask]
                    safe_indices = safe_df.index.tolist()
                    filtered_count = initial_count - len(safe_df)
                    logger.info(f" {allergy}: removed {filtered_count} dangerous recipes")
                else:
                    logger.warning(f" Column '{column_name}' not found - allergen not detected automatically")
        
        if safe_df.empty:
            logger.error("Alert: No safe recipes after filtering")
            return RecommendationResponse(
                recommendations=[],
                metrics=InferenceMetrics(
                    total_time_ms=(time.time() - start_time) * 1000,
                    embedding_time_ms=0,
                    similarity_time_ms=0,
                    processing_time_ms=0,
                    query_length=len(request.user_query),
                    total_recipes_in_db=len(df),
                    safe_recipes_after_filtering=0,
                    recipes_returned=0,
                    top_n=request.top_n,
                    avg_similarity_score=0,
                    max_similarity_score=0,
                    min_similarity_score=0,
                    allergies_filtered=request.allergies
                )
            )
        
        logger.info(f"Safe recipes available: {len(safe_df)}/{len(df)}")
        
        logger.info(f"Semantic ranking on safe recipes...")
        
        safe_embeddings = recipe_embeddings[safe_indices]
        
        # Measure embedding time
        embedding_start = time.time()
        query_embedding = model.encode([request.user_query], convert_to_numpy=True)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Measure similarity time
        similarity_start = time.time()
        similarities = cosine_similarity(query_embedding, safe_embeddings)[0]
        similarity_time = (time.time() - similarity_start) * 1000
        
        # Measure processing time
        processing_start = time.time()
        top_indices = np.argsort(similarities)[::-1][:request.top_n]
        
        recommendations = []
        for idx in top_indices:
            safe_recipe_idx = safe_indices[idx]
            recipe = df.iloc[safe_recipe_idx]
            
            recommendation = RecipeResponse(
                recipe_name=str(recipe.get('recipe_name', 'No name')),
                similarity_score=float(similarities[idx]),
                ingredients_text=str(recipe.get('ingredients_text', 'Not available')),
                instructions=str(recipe.get('instructions', 'Not available')),
                source=str(recipe.get('source', 'Unknown')),
                language=str(recipe.get('language', 'unknown'))
            )
            
            recommendations.append(recommendation)
        
        processing_time = (time.time() - processing_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        top_similarities = similarities[top_indices]
        
        metrics = InferenceMetrics(
            total_time_ms=round(total_time, 2),
            embedding_time_ms=round(embedding_time, 2),
            similarity_time_ms=round(similarity_time, 2),
            processing_time_ms=round(processing_time, 2),
            query_length=len(request.user_query),
            total_recipes_in_db=len(df),
            safe_recipes_after_filtering=len(safe_df),
            recipes_returned=len(recommendations),
            top_n=request.top_n,
            avg_similarity_score=round(float(np.mean(top_similarities)), 4),
            max_similarity_score=round(float(np.max(top_similarities)), 4),
            min_similarity_score=round(float(np.min(top_similarities)), 4),
            allergies_filtered=request.allergies
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            metrics=metrics
        )
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"ERROR después de {total_time:.2f}ms: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)