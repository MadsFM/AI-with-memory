from fastapi import FastAPI
import jokesterAI

app = FastAPI()

app.include_router(jokesterAI.router, prefix="/AI", tags=["Joey"])