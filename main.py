from fastapi import FastAPI
import ai

app = FastAPI()

app.include_router(jokesterAI.router, prefix="/AI", tags=["Joey"])