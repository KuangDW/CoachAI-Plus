interface DisplayRallyData {
  rally: number;
  ball_round: number;
  player: string;
  player_score: number;
  opponent_score: number;
  ball_location_x: number;
  ball_location_y: number;
  player_location_x: number;
  player_location_y: number;
  opponent_location_x: number;
  opponent_location_y: number;
  lose_reason: string;
}

interface DisplayRallyResponse {
  player: string;
  opponent: string;
  records: DisplayRallyData[];
}

interface Visualize {
  ShotType: string;
  top_win_states: string;
  top_lose_states: string;
}

interface VisualizeResponse {
  visualize: DisplayRallyResponse;
  [key: string]: Visualize;
}
