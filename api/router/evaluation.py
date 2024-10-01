from fastapi import APIRouter, UploadFile, Form
import pandas as pd
from typing import List
from typing_extensions import Annotated
from Shot_Evaluation.main import Evaluation

EvaluztionRouter = APIRouter()

@EvaluztionRouter.post("")
async def EvaluationAPI(last_ball_round: Annotated[int, Form()], files: List[UploadFile]):
  result = []
  for file in files:
      match = pd.read_csv(file.file)
      match['match_id'] = match['match_id'].astype(int).astype(str)
      match_id_mapping = {'1':'23', '3': '28', '5': '30', '6': '31', '7': '32', '13': '49', '2': '25', '4': '29', '8': '36', '9': '43', '10': '44',
                  '11': '45', '19': '55', '14': '50', '15': '51', '30': '72', '36': '79', '44': '97', '17': '53', '26': '64', '12': '48',
                  '20': '56', '23': '60', '28': '69', '41': '88', '18': '54', '16': '52', '27': '66', '33': '75', '37': '82', '24': '61',
                  '39': '86', '21': '57', '34': '76', '29': '71', '35': '78', '38': '85', '22': '58', '32': '74', '25': '63', '31': '73',
                  '43': '94', '42': '89', '40': '87'}
      match_id_mapping = {k: int(v) for k, v in match_id_mapping.items()}
      match['match_id'] = match['match_id'].map(match_id_mapping)
      file_dict = Evaluation(match, last_ball_round, "./api/statics")
      result.append(file_dict)

  return result