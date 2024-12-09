interface BasicStore {
  files: FileList | null;
  index: number;
  loading: boolean;
  setFiles: (files: FileList | null) => void;
  setIndex: (index: number) => void;
  setLoading: (loading: boolean) => void;
}

interface ShotStore extends BasicStore {
  data: EvaluationResponse[];
  n: number;
  setData: (data: EvaluationResponse[]) => void;
  setN: (n: number) => void;
}

interface TacticStore extends BasicStore {
  data: TacticResponse[];
  setData: (data: TacticResponse[]) => void;
}

interface VisualizeStore extends BasicStore {
  data: VisualizeResponse[];
  opponentType: boolean;
  playerLocationArea: boolean;
  opponentLocationArea: boolean;
  hitArea: boolean;
  displayIndex: number;
  timer: number | null;
  setOpponentType: (opponentType: boolean) => void;
  setPlayerLocationArea: (playerLocationArea: boolean) => void;
  setOpponentLocationArea: (opponentLocationArea: boolean) => void;
  setHitArea: (hitArea: boolean) => void;
  setData: (data: VisualizeResponse[]) => void;
  setDisplayIndex: (displayIndex: number) => void;
  setTimer: (timer: number | null) => void;
  addDisplayIndex: () => void;
  minusDisplayIndex: () => void;
}
