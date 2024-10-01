import sys, os

# Add paths
ROOT_FILDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_FILDER)
sys.path.append(f"{ROOT_FILDER}/api")
sys.path.append(f"{ROOT_FILDER}/Shot_Evaluation")
sys.path.append(f"{ROOT_FILDER}/Visualization_And_Win_Lose_Reason_Statistics")
sys.path.append(f"{ROOT_FILDER}/Tactic_Evaluation")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from router.evaluation import EvaluztionRouter
from router.visualize import VisualizeRouter
from router.tactic import TacticRouter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(EvaluztionRouter, prefix="/api/evaluation")
app.include_router(VisualizeRouter, prefix="/api/visualize")
app.include_router(TacticRouter, prefix="/api/tactic")

app.mount("/statics", StaticFiles(directory=f"{os.path.dirname(os.path.abspath(__file__))}/statics"), name="images")
app.mount("/", StaticFiles(directory=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/frontend/dist", html=True), name="spa")




