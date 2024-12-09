import pandas as pd
import numpy as np
from fastapi import UploadFile

def basicInfo(match: pd.DataFrame, file: UploadFile):
  players = match["player"].unique().astype(str)
  return {
    "filename": file.filename,
    "rally": match["rally_id"].nunique(),
    "players": [x for x in players if x != 'nan']
  }