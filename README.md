# Job_Recommender
Job recommendation app

# Learning
- What is the intuition behind cross encoders?
    -  When running retrieval systems, if we only use a bi-encoder, there will be information loss since the query and document are encoded separately. When documents in bi-encoders are embedded as vectors, the document loses: 
        - Exact wording relationships
        - Fine-grained token interactions
        - Nuanced relevance can get blurred
    - For example the query "python engineer with retrieval experience" when compared to a document A "Built semantic search and RAG systems in Python" and a document B "Python backend engineer for payment systems", a bi-encoder might think both A and B are close due to mentions of python and engineering jobs even though document A is a way better match.
    - Cross encoders will process both the query and document together

- Pydantic request model
    - When people call the FastAPI API, FastAPI needs to know what the JSON should look like. Pydantic request model sets this up. For this project, it uses the "BaseModel" from the "pydantic" library.
    - FastAPI will automatically reads the incoming JSON request, validate if it matches any of the defined models, and passes it to the endpoint as a "SearchRequest" object. A 422 error will be raised if there were no matching endpoints.

- FastAPI
    - FastAPI allows loading items with heavy initial loads at the start of the server and saves them in an app.state for later use

- Docker
    - **What is Docker?** Docker is a containerization software that packages an app and all its dependencies so it runs consistently regardless of the host machine.
    - **What is a Dockerfile?** A Dockerfile is a recipe file that describes how to build a Docker image including base OS, dependencies to install, and the command to start the app.
    - **What is docker-compose?** Docker-compose is a config file that defines and orchestrates multiple services (containers) together. This project used two services to keep the FastAPI backend and Streamlit frontend separate.
    - **Why can't containers use localhost?** Each container has its own isolated network, so `localhost` inside a container refers to that container itself - not the host machine or other containers. Docker Compose creates a shared network between services and assigns each one a DNS name matching its service name. So the Streamlit container reaches the API at `http://api:8000` instead of `http://127.0.0.1:8000`.

# Resources
Sentence Transformers: https://www.geeksforgeeks.org/nlp/sentence-transformer/