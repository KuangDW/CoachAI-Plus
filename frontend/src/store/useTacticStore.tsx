import { create } from "zustand";

const useTacticStore = create<TacticStore>((set) => ({
  files: null,
  n: 3,
  index: 0,
  loading: false,
  data: [],
  setFiles: (f) => set(() => ({ files: f })),
  setIndex: (i) => set(() => ({ index: i })),
  setLoading: (l) => set(() => ({ loading: l })),
  setData: (d) => set(() => ({ data: d })),
}));

export default useTacticStore;
