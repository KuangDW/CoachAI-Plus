from fastapi import APIRouter, File, UploadFile
import pandas as pd
from typing import List
from Tactic_Evaluation.main import Tactic

TacticRouter = APIRouter()

@TacticRouter.post("")
async def TacticAPI(files: List[UploadFile]):
  result = []
  for file in files:
      match = pd.read_csv(file.file)
      file_dict = Tactic(match, "./api/statics")
      result.append(file_dict)

  return result