interface Tactic {
  pie_chart: string;
  [key: string]: {
    histogram: string;
    coord_diagram: string;
  };
}

interface TacticResponse {
  [key: string]: Tactic;
}
