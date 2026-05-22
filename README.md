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

# Resources
Sentence Transformers: https://www.geeksforgeeks.org/nlp/sentence-transformer/