from fastapi import FastAPI, Depends
from app.routers import routes_gemini
from chainlit.utils import mount_chainlit

app = FastAPI()
app.include_router(routes_gemini.router, prefix="/api", tags=["Gemini"])


mount_chainlit(
    app=app,
    target="chainlit_app.py",
    path="/chainlit",
)