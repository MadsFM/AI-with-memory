from fastapi import FastAPI
import AI

app = FastAPI()


app.include_router(jokesterAI.router, prefix="/AI", tags=["It-Expert"])
