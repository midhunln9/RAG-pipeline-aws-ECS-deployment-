from fastapi import FastAPI
from rag_pipeline.api.routes.ask_endpoint import ask_endpoint
import uvicorn

app = FastAPI()

app.include_router(ask_endpoint)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
