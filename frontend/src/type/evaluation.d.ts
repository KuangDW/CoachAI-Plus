interface Evaluation {
  shot_type: string[];
  last_ball: {
    Smash: string;
    "Push Shot": string;
    Lob: string;
    "Smash Defence": string;
    "Net Shot": string;
    Drop: string;
    Clear: string;
    Drive: string;
  };
  top_reasons: {
    win: string;
    loose: string;
  };
  shot_influence: string[];
}

interface DensityDifference {
  Clear_win: string;
  Clear_lose: string;
  "Push Shot_win": string;
  "Push Shot_lose": string;
  Smash_win: string;
  Smash_lose: string;
  "Smash Defence_win": string;
  "Smash Defence_lose": string;
  Drive_win: string;
  Drive_lose: string;
  "Net Shot_win": string;
  "Net Shot_lose": string;
  Lob_win: string;
  Lob_lose: string;
  Drop_win: string;
  Drop_lose: string;
}

interface EvaluationResponse {
  energetic_cost: string;
  filename: string;
  players: string[];
  rally: number;
  density_difference?: DensityDifference;
  [key: string]: Evaluation;
}
