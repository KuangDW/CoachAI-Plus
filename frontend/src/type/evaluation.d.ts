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

interface EvaluationResponse {
  energetic_cost: string;
  [key: string]: Evaluation;
}
