from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.search_jobs import search
from app.retrieve import load_artifacts, load_model


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs, job_embeddings = load_artifacts()
    model = load_model()

    app.state.jobs = jobs
    app.state.job_embeddings = job_embeddings
    app.state.model = model

    yield

    # Shutdown / Clean resources here if necessary


app = FastAPI(lifespan=lifespan)


@app.post("/search")
def search_endpoint(request: Request, body: SearchRequest):
    return search(body.query, request.app.state.model, request.app.state.jobs, 
                  request.app.state.job_embeddings, body.top_k)