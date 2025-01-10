from typing import List, Optional, Dict, Any, Tuple
from config.settings import get_settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
import torch
import psycopg2
import datetime
import time
import requests

class VectorDB:
    """A class for managing vector operations and database interactions."""

    def __init__(self, model: str = "local"):
        """Initialize the VectorDB with settings and  model."""
        self.settings = get_settings()
        self.embedding_model = self.settings.model.path_model
        self.vector_settings = self.settings.vector_db
        self.conn = psycopg2.connect(self.settings.database.service_url)
        self.cursor = self.conn.cursor()
        
        if model == "local":
            if not hasattr(self, 'model'):
                self.model = SentenceTransformer(self.settings.model.path_model)
                self.model.to(self.settings.model.device)
                

        
    def get_embedding_hugging_face(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using Hugging Face model.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        
        text = text.replace("\n", " ")
        text_list = [text]
        
        start_time = time.time()
        
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.settings.model.path_model}"
        headers = {
            "Authorization": f"Bearer {self.settings.model.hugging_face_token}"
        }
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            embedding = response.json()
            
            elapsed_time = time.time() - start_time
            logging.info(f"Hugging-face embedding generated in {elapsed_time:.3f} seconds")
            
            return embedding
        
        else:
            raise Exception(f"Error generating embedding: {response.text}")
        
    def get_embedding_local(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using local model.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        
        
        text = text.replace("\n", " ")
        
        start_time = time.time()
        
        with torch.no_grad():
            embedding = self.model.encode(
                [text],
                convert_to_tensor=True,
                device=self.settings.model.device,
                batch_size=32,
                show_progress_bar=False
            )
            embedding = embedding[0].cpu().numpy().tolist()
        
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        
        return embedding

    def get_embedding_openAI(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self):
        """Create the embeddings table."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.vector_settings.table_name} (
            id SERIAL PRIMARY KEY,
            title TEXT,
            metadata JSONB,
            contents TEXT,
            embedding VECTOR({self.vector_settings.embedding_dimensions})
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def create_index(self):
        """Create an index on the embedding column."""
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS idx_embedding
        ON {self.vector_settings.table_name} USING ivfflat (embedding);
        """
        self.cursor.execute(create_index_query)
        self.conn.commit()

    def upsert(self, records_df: pd.DataFrame):
        """Upsert records into the embeddings table."""
        for _, row in records_df.iterrows():
            upsert_query = f"""
            INSERT INTO {self.vector_settings.table_name} (title, metadata, contents, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title,
                metadata = EXCLUDED.metadata,
                contents = EXCLUDED.contents,
                embedding = EXCLUDED.embedding;
            """
            self.cursor.execute(upsert_query, (row['title'],row['metadata'], row['contents'], row['embedding']))
        self.conn.commit()

    def _format_time_range(self, time_range: Tuple[int, int]) -> Tuple[str, str]:
        start_date = pd.to_datetime(f"{time_range[0]}-01-01").tz_localize('UTC').strftime('%Y-%m-%d')
        end_date = pd.to_datetime(f"{time_range[1]}-12-31").tz_localize('UTC').strftime('%Y-%m-%d')
        return start_date, end_date
    
    def search(
        self,
        conn,
        query_text: str,
        metadata: Optional[Dict] = None,
        limit: int = 16,
        popularity_weight: float = 0.05,  
        model: str= "local"
    ) -> List[Tuple]:
        """
        Search for similar movies with popularity weighting and optional filters.
        
        Args:
            conn: Active psycopg2 connection
            query_text: Text to embed and compare
            limit: Max results
            metadata: Metadata filters (genres, keywords, original_language, time_range)
            popularity_weight: Weight given to popularity in final score (0.0 to 1.0)
        Returns:
            List of result tuples (id, title, metadata, contents, similarity)
        """
        if model == "local":
            query_embedding = self.get_embedding_local(query_text)
        elif model == "hugging-face":
            query_embedding = self.get_embedding_hugging_face(query_text)
        elif model == "openAI":
            query_embedding = self.get_embedding_openAI(query_text)    
        
        query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Modified SQL query to include popularity in scoring
        sql_query = """
        WITH weighted_results AS (
            SELECT
                id,
                title,
                metadata,
                contents,
                1 - (embedding <=> %s::vector) AS similarity,
                COALESCE(CAST(metadata->>'popularity' AS FLOAT), 0) AS popularity,
                -- Normalize popularity to 0-1 range within the filtered result set
                CASE 
                    WHEN MAX(CAST(metadata->>'popularity' AS FLOAT)) OVER() = MIN(CAST(metadata->>'popularity' AS FLOAT)) OVER()
                    THEN 0.5
                    ELSE (CAST(metadata->>'popularity' AS FLOAT) - MIN(CAST(metadata->>'popularity' AS FLOAT)) OVER()) / 
                        NULLIF(MAX(CAST(metadata->>'popularity' AS FLOAT)) OVER() - MIN(CAST(metadata->>'popularity' AS FLOAT)) OVER(), 0)
                END AS normalized_popularity
            FROM
                embeddings
            WHERE
                TRUE
        """
        
        params = [query_vector_str]
        
        # Add metadata filters
        if metadata:
            for key, value in metadata.items():
                if value is None or value == '':
                    continue
                
                if key in ('genres', 'keywords'):
                    terms = [term.strip() for term in value.split(',')]
                    if terms:
                        conditions = []
                        for term in terms:
                            conditions.append(f"metadata->>'{key}' ILIKE %s")
                            params.append(f"%{term}%")
                        sql_query += f" AND ({' OR '.join(conditions)})"
                
                elif key == 'original_language':
                    sql_query += " AND metadata->>'original_language' = %s"
                    params.append(value)
                
                elif key == 'time_range':
                    start_date, end_date = self._format_time_range(value)
                    print(start_date, end_date)
                    sql_query += " AND (metadata->>'release_date')::timestamp BETWEEN %s AND %s"
                    params.extend([start_date, end_date])
        
        # Close the CTE and add the final scoring and ordering
        sql_query += """
        )
        SELECT
            id,
            title,
            metadata,
            contents,
            similarity,
            -- Calculate final score combining similarity and normalized popularity
            (similarity * (1 - %s) + normalized_popularity * %s) as final_score
        FROM
            weighted_results
        ORDER BY
            final_score DESC
        LIMIT %s
        """
        
        params.extend([popularity_weight, popularity_weight, limit])
        
        with conn.cursor() as cur:
            cur.execute(sql_query, params)
            results = cur.fetchall()
        
        return results