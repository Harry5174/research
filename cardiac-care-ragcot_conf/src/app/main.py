import os
from app.routers.routes_main import app
import uvicorn
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    uvicorn.run(app=app, host="localhost", port=8000, log_level="info")