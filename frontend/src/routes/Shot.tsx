import { useState } from "react";
import { fetchApi } from "../utils/api";
import EvaluationDisplay from "../components/EvaluationDisplay";

const Evaluation = () => {
  const [files, setFiles] = useState<FileList | null>(null);
  const [n, setN] = useState<number>(3);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<EvaluationResponse[]>([]);
  const [index, setIndex] = useState(0);

  const handleReset = () => {
    setIndex(0);
    setData([]);
  };

  const handleChangeFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFiles(e.target.files);
  };

  const handleSubmit = async (e: React.ChangeEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!files) return;
    setLoading(true);
    const form = new FormData();
    form.append("last_ball_round", n.toString());
    for (let i = 0; i < files.length; i++) {
      form.append("files", files[i]);
    }
    try {
      const res = await fetchApi("/evaluation", "POST", {
        data: form,
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "application/json",
        },
      });
      setData(res.data as EvaluationResponse[]);
      setLoading(false);
    } catch (e) {
      console.error(e);
      setLoading(false);
    }
  };

  return (
    <>
      <div className="flex gap-2">
        <div className="grow overflow-hidden">
          <div className="flex flex-col gap-4 overflow-auto">
            <div className="flex justify-between items-center mx-8">
              <div className="flex flex-col gap-2">
                <h1 className="text-4xl font-bold">Shot Evaluation</h1>
                <p>Evaluate player in badminton match</p>
              </div>
              {data.length ? (
                <button className="btn btn-secondary" onClick={handleReset}>
                  BACK
                </button>
              ) : (
                ""
              )}
            </div>
            {data.length ? (
              <div className="card bg-base-100 shadow-xl m-6 p-6">
                <div className="flex justify-between">
                  <button
                    className="btn btn-primary"
                    disabled={index < 1}
                    onClick={() => setIndex((i) => i - 1)}
                  >
                    Prev
                  </button>
                  <h1 className="text-2xl font-bold">Set {index + 1}</h1>
                  <button
                    className="btn btn-primary"
                    disabled={index > data.length - 2}
                    onClick={() => setIndex((i) => i + 1)}
                  >
                    Next
                  </button>
                </div>
                {<EvaluationDisplay data={data[index]} />}
              </div>
            ) : (
              <form className="flex flex-col gap-4 " onSubmit={handleSubmit}>
                <div className="card bg-base-100 shadow-xl mx-4">
                  <div className="card-body gap-4">
                    <h2 className="card-title">Upload Files</h2>
                    <input
                      onChange={handleChangeFiles}
                      type="file"
                      multiple
                      accept=".csv"
                      className="file-input file-input-bordered file-input-sm w-full max-w-xs"
                    />
                    <h2 className="card-title">Last ball round</h2>
                    <input
                      type="number"
                      value={n}
                      onChange={(e) => setN(parseInt(e.target.value))}
                      placeholder="last ball round"
                      className="input input-bordered w-full max-w-xs"
                    />
                  </div>
                </div>

                <button className="btn btn-primary mx-4" type="submit">
                  Analysis
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
      {loading ? (
        <div className="w-screen h-screen z-10 bg-black absolute top-0 left-0 bg-opacity-50 flex items-center justify-center">
          <span className="loading loading-spinner w-16"></span>
        </div>
      ) : (
        ""
      )}
    </>
  );
};

export default Evaluation;
