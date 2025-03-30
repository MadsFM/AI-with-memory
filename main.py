from fastapi import FastAPI
import ai

app = FastAPI()


app.include_router(ai.router, prefix="/AI", tags=["It-Expert"])
