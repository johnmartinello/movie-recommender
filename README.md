# Movie Recommendation System
### NLP movie recommendation
This project is a movie recommendation system that leverages vector embeddings and similarity search to find movies similar to a given Natural Language query. The system processes a large dataset of movie information, generates embeddings for movie descriptions, and stores these embeddings in a PostgreSQL database with pgvector. The project includes functionalities for data cleaning, embedding generation, batch processing, and similarity search. I'm building this project on top of Dave Ebbelaar youtube tutorial ([link](https://www.youtube.com/watch?v=hAdEuDBN57g)). So if there is any problems, his video will be more helpfull for the initial database setup and such.

## Website
I made a website to showcase how this would look in a real product ([youtube](https://youtu.be/stPRuYNDRRw):
![Movie Recommender](video.gif)


## Prerequisites
Before you start, make sure you have the following installed:

1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **PostgreSQL with pgvector extension**: This will be set up using Docker.
3. **Python with pip**: I'm using python version 3.12.8
3. **GPU**: If you are running the model locally and want to leverage GPU acceleration, ensure you have a compatible GPU and CUDA installed.
4. **Hugging Face API Token**: If you are using the Hugging Face model, you will need an API token. [API token](https://huggingface.co/settings/tokens)
5. **Movie Dataset**: Download the movie dataset from [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies).

## Step-by-Step Instructions
### 1. Create `.env` File
Create a `.env` file in the root directory of the project with the following content:
- SERVICE_URL=postgres://postgres:123@localhost:5432/postgres 
- HUGGING_FACE_TOKEN=your_hugging_face_token

### 2. Download packages
Download packages from `requirements.txt` with command:
```
pip install requirements.txt
```
### 3. Run Docker Compose to Create pgvector Database Server
Navigate to the `docker` directory and run the following command to start the pgvector database server:
```
docker-compose up -d
```

### 4. Populate the Database
Run the `insert_vectors.py` script to load the movie dataset, generate embeddings, and populate the database:
```
python app/insert.vectors.py
```
Make sure the path to the movie dataset CSV file is correctly set in the script.

### 5. Start the Server
Run the `main.py` script to start the FastAPI server:
```
uvicorn app.main:app --reload
```
The API will be available at http://localhost:8000.

### API Endpoint
- POST /api/movieFinder/search: Search for similar movies based on a query and optional metadata filters
#### Example Request
```
{
    "query": "A young jazz drummer is trying to become one of the best.",
    "keywords": "drama, music",
    "language": "en",
    "yearFrom": "1960",
    "yearTo": "2020",
    "selectedGenres": ["Drama", "Music"]
}
```
#### Example Response (id,title,metadata[],similarity, final_score)
```
[
    136,
    "Whiplash",
    {
        "genres": "Drama, Music",
        "keywords": "new york city, concert, jazz, obsession, music teacher, conservatory, montage, public humiliation, jazz band, young adult, music school, based on short film",
        "movie_id": "244786",
        "popularity": 54.495,
        "poster_path": "/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "release_date": "2014-10-10",
        "original_language": "en",
        "production_companies": "Bold Films, Blumhouse Productions, Right of Way Films, Sierra/Affinity, Stage 6 Films, Sony Pictures Classics",
        "production_countries": "United States of America"
    },
    "Under the direction of a ruthless instructor, a talented young drummer begins to pursue perfection at any cost, even his humanity.",
    0.5044895711237017,
    0.4556962883438634
]
```

## Main Files Explanation


### `docker/docker-compose.yml`
Defines the Docker services for setting up the PostgreSQL database with pgvector extension.

### `app/config/settings.py`
Contains the configuration settings for the application, including model settings, database settings, and vector database settings.

### `app/insert.vectors.py`
Script to load the movie dataset, clean the data, generate embeddings, and populate the database.

### `app/database/vector_db.py`
Contains the `VectorDB` class for managing vector operations and database interactions, including embedding generation and similarity search.

### `app/similarity_search.py`
Contains the `Search` class for performing similarity searches on the movie dataset using the vector database.

### `app/main.py`
Defines the FastAPI application and API endpoints for searching similar movies based on a query and metadata filters.
