from fastapi import APIRouter, Form, UploadFile
import pandas as pd
import numpy as np
import csv, codecs
from typing import List
from typing_extensions import Annotated
from Visualization_And_Win_Lose_Reason_Statistics.main import Visualize

VisualizeRouter = APIRouter()

@VisualizeRouter.post("")
async def VisualizeAPI(opponent_type: Annotated[bool, Form()] , player_location_area: Annotated[bool, Form()] , opponent_location_area: Annotated[bool, Form()] , hit_area: Annotated[bool, Form()] ,files: List[UploadFile]):
  result = []
  state = (opponent_type, player_location_area, opponent_location_area, hit_area)
  for file in files:
      match = pd.read_csv(file.file)
      file_dict = Visualize(match, state, "./api/statics")

      players = match["player"].unique().tolist()
      if(len(players) != 2):
        raise KeyError("There are not 2 players")
      match = match.filter(items=['rally', 'ball_round', 'player', 'player_score', 'opponent_score', 'landing_x', 'landing_y', 'hit_x', 'hit_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'lose_reason'])
      player = players[0]
      opponent = players[1]
      if match["player"][0] == opponent:
        tmp = player
        player = opponent
        opponent = tmp
      match["ball_location_x"] = np.where(match["player"] == player, match["hit_x"] * 177.5, match["hit_x"] * -177.5)
      match["ball_location_y"] = np.where(match["player"] == player, 240 - match["hit_y"] * 240, -240 + match["hit_y"] * 240)
      match["player_location_x"] = np.where(match["player"] == player, match["hit_x"] * 177.5, match["opponent_location_x"] * -177.5)
      match["player_location_y"] = np.where(match["player"] == player, 240 - match["hit_y"] * 240, -240 + match["opponent_location_y"] * 240)
      match["opponent_location_x"] = np.where(match["player"] == opponent, match["hit_x"] * 177.5, match["opponent_location_x"] * -177.5)
      match["opponent_location_y"] = np.where(match["player"] == opponent, 240 - match["hit_y"] * 240, -240 + match["opponent_location_y"] * 240)

      match["player_location_x"] = np.where(match["player"] == player, match["player_location_x"], match["player_location_x"] * -1)
      match["player_location_y"] = np.where(match["player"] == player, match["player_location_y"], match["player_location_y"] * -1)
      match["opponent_location_x"] = np.where(match["player"] == player, match["opponent_location_x"], match["opponent_location_x"] * -1)
      match["opponent_location_y"] = np.where(match["player"] == player, match["opponent_location_y"], match["opponent_location_y"] * -1)

      match = match.filter(items=['rally', 'ball_round', 'player', 'player_score', 'opponent_score', 'ball_location_x', 'ball_location_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'lose_reason'])
      
      match = match[match['player_score'].notnull()]
      match = match.fillna("")

      file_dict["visualize"] = {"player": player, "opponent": opponent, "records": list(match.to_dict(orient="records"))}

      result.append(file_dict)

  return result

@VisualizeRouter.post("/new")
async def test(files: List[UploadFile]):
  result = []
  for file in files:
    match = pd.read_csv(file.file)
    players = match["player"].unique().tolist()
    if(len(players) != 2):
       raise KeyError("There are not 2 players")
    match = match.filter(items=['rally', 'ball_round', 'player', 'player_score', 'opponent_score', 'landing_x', 'landing_y', 'hit_x', 'hit_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'lose_reason'])
    player = players[0]
    opponent = players[1]
    if match["player"][0] == opponent:
       tmp = player
       player = opponent
       opponent = tmp
    match["ball_location_x"] = np.where(match["player"] == player, match["hit_x"] * 177.5, match["hit_x"] * -177.5)
    match["ball_location_y"] = np.where(match["player"] == player, 240 - match["hit_y"] * 240, -240 + match["hit_y"] * 240)
    match["player_location_x"] = np.where(match["player"] == player, match["hit_x"] * 177.5, match["opponent_location_x"] * -177.5)
    match["player_location_y"] = np.where(match["player"] == player, 240 - match["hit_y"] * 240, -240 + match["opponent_location_y"] * 240)
    match["opponent_location_x"] = np.where(match["player"] == opponent, match["hit_x"] * 177.5, match["opponent_location_x"] * -177.5)
    match["opponent_location_y"] = np.where(match["player"] == opponent, 240 - match["hit_y"] * 240, -240 + match["opponent_location_y"] * 240)

    match["player_location_x"] = np.where(match["player"] == player, match["player_location_x"], match["player_location_x"] * -1)
    match["player_location_y"] = np.where(match["player"] == player, match["player_location_y"], match["player_location_y"] * -1)
    match["opponent_location_x"] = np.where(match["player"] == player, match["opponent_location_x"], match["opponent_location_x"] * -1)
    match["opponent_location_y"] = np.where(match["player"] == player, match["opponent_location_y"], match["opponent_location_y"] * -1)

    match = match.filter(items=['rally', 'ball_round', 'player', 'player_score', 'opponent_score', 'ball_location_x', 'ball_location_y', 'player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'lose_reason'])
    
    match = match[match['player_score'].notnull()]
    match = match.fillna("")
    result.append({"player": player, "opponent": opponent, "record": list(match.to_dict(orient="records"))})
  return result
