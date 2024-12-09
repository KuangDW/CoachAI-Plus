import sys, os

# Add paths
ROOT_FILDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_FILDER)
sys.path.append(f"{ROOT_FILDER}/BadmintonEnv")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from router import EvaluztionRouter, VisualizeRouter, TacticRouter, RunEnvRouter
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(EvaluztionRouter, prefix="/api/evaluation")
app.include_router(TacticRouter, prefix="/api/tactic")
app.include_router(VisualizeRouter, prefix="/api/visualize")
app.include_router(RunEnvRouter, prefix="/api/runenv")

app.mount("/statics", StaticFiles(directory=f"{os.path.dirname(os.path.abspath(__file__))}/statics"), name="images")
app.mount("/", StaticFiles(directory=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/frontend/dist", html=True), name="spa")

if __name__ == '__main__':
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)


