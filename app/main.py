from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from similarity_search import Search
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Recommender API")
router = APIRouter()
search_db = Search()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request validation
class MovieSearchRequest(BaseModel):
    query: str
    keywords: Optional[str] = None
    language: Optional[str] = None
    yearFrom: Optional[str] = None
    yearTo: Optional[str] = None
    selectedGenres: Optional[List[str]] = []

@router.post("/api/movieFinder/search")  
async def search_movies(search_params: MovieSearchRequest):
    try:

        if not search_params.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        time_start = search_params.yearFrom
        time_end = search_params.yearTo
        
        if time_start == "":
            time_start = "1900"
        if time_end == "":
            time_end = "2025"
            
        time_range = (time_start, time_end)
        #check if the time range is valid
        if time_range[0] > time_range[1]:
            time_range = (time_range[1], time_range[0])
        
        metadata = {
            "keywords": search_params.keywords,
            "original_language": search_params.language,
            "time_range": time_range,
            "genres": search_params.selectedGenres
        }

        metadata = {k: v for k, v in metadata.items() if v is not None}

        results = search_db.search_movies(
            query=search_params.query,
            metadata=metadata,
            limit=128,
            popularity_weight=0.1,
            model="hugging-face"
        )

        return results

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
