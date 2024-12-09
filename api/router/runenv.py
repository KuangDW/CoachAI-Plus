from fastapi import APIRouter
from pydantic import BaseModel

from BadmintonEnv.RunEnv import main
RunEnvRouter = APIRouter()

class RunEnvBody(BaseModel):
    player1: str;
    player2: str;

@RunEnvRouter.post("")
async def RunEnvAPI(data: RunEnvBody):
  data = main(data.player1, data.player2)

  return data
