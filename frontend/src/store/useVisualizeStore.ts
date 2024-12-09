import { create } from "zustand";

const useVisualizeStore = create<VisualizeStore>((set) => ({
  files: null,
  index: 0,
  loading: false,
  data: [],
  opponentType: true,
  playerLocationArea: true,
  opponentLocationArea: true,
  hitArea: true,
  displayIndex: 0,
  timer: null,
  setFiles: (f) => set(() => ({ files: f })),
  setIndex: (i) => set(() => ({ index: i })),
  setLoading: (l) => set(() => ({ loading: l })),
  setData: (d) => set(() => ({ data: d })),
  setOpponentType: (v) => set(() => ({ opponentType: v })),
  setPlayerLocationArea: (v) => set(() => ({ playerLocationArea: v })),
  setOpponentLocationArea: (v) => set(() => ({ opponentLocationArea: v })),
  setHitArea: (v) => set(() => ({ hitArea: v })),
  setDisplayIndex: (v) => set(() => ({ displayIndex: v })),
  setTimer: (v) => set(() => ({ timer: v })),
  addDisplayIndex: () => set((state) => ({ displayIndex: state.displayIndex + 1 })),
  minusDisplayIndex: () => set((state) => ({ displayIndex: state.displayIndex - 1 })),
}));

export default useVisualizeStore;
