from typing import Dict, Optional
import psycopg2
import logging
from database.vector_db import VectorDB
import logging

logger = logging.getLogger(__name__)

class Search:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="123",
            host="localhost",
            port="5432"
            )
        self.vector_db = VectorDB(model="hugging-face")

    def normalize_metadata(self, metadata):
        """Normalize metadata input formats"""
        def map_language(input):
            language_map = {
                "English": "en",
                "Spanish": "es",
                "French": "fr",
                "german": "de",
                "italian": "it",
                "japanese": "ja",
                "korean": "ko",
                "chinese": "zh",
                "hindi": "hi",
                "arabic": "ar",
                "russian": "ru",
                "portuguese": "pt",
                "turkish": "tr",
                "dutch": "nl",
                "polish": "pl",
                "swedish": "sv",
                "danish": "da",
                "norwegian": "no",
                "finnish": "fi"
                }
            return language_map.get(input, input)
    

        normalized = {}
        
        # Handle genres
        if "genres" in metadata:
            if isinstance(metadata["genres"], list):
                normalized["genres"] = ",".join(metadata["genres"])
            else:
                normalized["genres"] = metadata["genres"]
        
        # Handle time_range
        if "time_range" in metadata:
            normalized["time_range"] = metadata["time_range"]
                
        for key in ["original_language", "keywords"]:
            if key in metadata:
                normalized[key] = metadata[key]
                
        if "original_language" in metadata:
            normalized["original_language"] = map_language(metadata["original_language"])
            
                
        return normalized
    
    def search_movies(
    self,
    query,
    metadata=None, 
    limit=16,
    popularity_weight=0.05,
    model="local"
):
        if not query:
            raise ValueError("Query cannot be empty")
            
        try:
            # Normalize metadata before processing
            normalized_metadata = self.normalize_metadata(metadata or {})
            
            # Ensure time_range is a tuple
            if "time_range" in normalized_metadata and not isinstance(normalized_metadata["time_range"], tuple):
                raise ValueError("time_range must be a tuple")
            
            logger.info(f"Searching with query: {query}, metadata: {normalized_metadata}")
            
            return self.vector_db.search(
                conn=self.conn,
                query_text=query,
                metadata=normalized_metadata,
                limit=limit,
                popularity_weight=popularity_weight,
                model=model
            )
                    
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise ValueError(f"Search failed: {str(e)}")
        
def main():
    search = Search()
    query = "A young jazz drummer is trying to become the one of the bests."
    metadata = {
        "genres": "Drama",
        "original_language": "en",
        "time_range": "1960-2020"
    }
    
    results = search.search_movies(query,metadata)
    print(results)


if __name__ == "__main__":
    pass
