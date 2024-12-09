import { create } from "zustand";

const useShotStore = create<ShotStore>((set) => ({
  files: null,
  n: 3,
  index: 0,
  loading: false,
  data: [],
  setFiles: (f) => set(() => ({ files: f })),
  setN: (n) => set(() => ({ n: n })),
  setIndex: (i) => set(() => ({ index: i })),
  setLoading: (l) => set(() => ({ loading: l })),
  setData: (d) => set(() => ({ data: d })),
}));

export default useShotStore;
