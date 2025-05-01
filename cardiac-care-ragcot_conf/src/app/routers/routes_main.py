from fastapi import FastAPI, Depends
from app.routers import routes_gemini

app = FastAPI()
app.include_router(routes_gemini.router, prefix="/api", tags=["Gemini"])