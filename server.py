from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from agent.functions import AgentFunstions

app = FastAPI(title="AI Estimator", version="0.0.1")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an instance of the AgentFunstions class
agent_functions = AgentFunstions()

@app.get("/")
async def root():
    return {"Server": "Running"}

# show the UI
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("front/front2.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: HTML file not found.</h1>"

# User query
@app.post("/query")
async def query(user_input: str):
    try:
        agent_process_response = await agent_functions.query_process(user_input)
        if agent_process_response["status"] != 0:
            return {"status": -1, "message": agent_process_response["message"], "data": {}}
        
        return {"status": 0, "message": "", "data": agent_process_response["data"]}
    except Exception as e:
        return {"status": -1, "message": str(e), "data": {}}
    

@app.post("/query/stream")
async def query_stream(user_input: str):
    try:
        generator = agent_functions.query_process_stream(user_input)
        return StreamingResponse(generator, media_type="application/x-ndjson")
    except Exception as e:
        return {"status": -1, "message": str(e), "data": {}}