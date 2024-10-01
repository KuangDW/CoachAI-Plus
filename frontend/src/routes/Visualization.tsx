import { useState } from "react";
import { fetchApi } from "../utils/api";
import VisualizeDisplay from "../components/VisualizeDisplay";
import Court from "../components/Court";

const Visualize = () => {
  const [files, setFiles] = useState<FileList | null>(null);
  const [opponentType, setOpponentType] = useState(true);
  const [playerLocationArea, setPlayerLocationArea] = useState(true);
  const [opponentLocationArea, setOpponentLocationArea] = useState(true);
  const [hitArea, setHitArea] = useState(true);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<VisualizeResponse[]>([]);
  const [index, setIndex] = useState(0);

  const [displayIndex, setDisplayIndex] = useState(0);
  const [timer, setTimer] = useState<null | number>(null);

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
    form.append("opponent_type", opponentType ? "true" : "false");
    form.append("player_location_area", playerLocationArea ? "true" : "false");
    form.append(
      "opponent_location_area",
      opponentLocationArea ? "true" : "false"
    );
    form.append("hit_area", hitArea ? "true" : "false");
    for (let i = 0; i < files.length; i++) {
      form.append("files", files[i]);
    }
    try {
      const res = await fetchApi("/visualize", "POST", {
        data: form,
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "application/json",
        },
      });
      setData(res.data as VisualizeResponse[]);
      setLoading(false);
    } catch (e) {
      console.error(e);
      setLoading(false);
    }
  };

  const nextBall = () => {
    if (displayIndex + 1 < data[index].visualize.records.length)
      setDisplayIndex((v) => v + 1);
  };

  const prevBall = () => {
    if (displayIndex > 0) setDisplayIndex((v) => v - 1);
  };

  const startTimer = () => {
    if (displayIndex + 1 < data[index].visualize.records.length) {
      const t = setInterval(nextBall, 1000);
      setTimer(t);
    }
  };

  const clearTimer = () => {
    if (timer) {
      clearInterval(timer);
      setTimer(null);
    }
  };

  return (
    <>
      <div className="flex flex-col gap-4 overflow-auto h-full">
        <div className="flex justify-between items-center mx-8 shrink-0">
          <div className="flex flex-col gap-2">
            <h1 className="text-4xl font-bold">
              Visualization & Winning/Losing Reason Statistics
            </h1>
            <p>
              Visualize badminton match content through animation to analyze the
              relationship between winning and losing states and behavior,
              identifying strengths and weaknesses to provide training insights
              for athletes.
            </p>
          </div>
          {data.length ? (
            <button className="btn btn-secondary" onClick={handleReset}>
              BACK
            </button>
          ) : (
            ""
          )}
        </div>
        <div className="grow">
          {data.length ? (
            <div className="flex gap-2">
              <div className="card gap-4 bg-base-100 shadow-xl mx-6 mb-6">
                <div className="flex justify-between p-6">
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
                <div className="px-6">
                  {<VisualizeDisplay data={data[index]} />}
                </div>

                {data.length ? (
                  <div>
                    <div className="flex flex-col divide-y divide-base-300 bg-secondary bg-opacity-50">
                      <div className="grid grid-cols-2 divide-x divide-base-300">
                        <div className="stat">
                          <div className="stat-title">
                            {data[index].visualize.player}
                          </div>
                          <div className="stat-value">
                            {
                              data[index].visualize.records[displayIndex]
                                .player_score
                            }
                          </div>
                          {/* <div className="stat-desc"></div> */}
                        </div>
                        <div className="stat">
                          <div className="stat-title">
                            {data[index].visualize.opponent}
                          </div>
                          <div className="stat-value">
                            {
                              data[index].visualize.records[displayIndex]
                                .opponent_score
                            }
                          </div>
                          {/* <div className="stat-desc"></div> */}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 divide-x divide-base-300">
                        <div className="stat">
                          <div className="stat-title">Rally</div>
                          <div className="stat-value">
                            {data[index].visualize.records[displayIndex].rally}
                          </div>
                          {/* <div className="stat-desc"></div> */}
                        </div>

                        <div className="stat">
                          <div className="stat-title">Ball</div>
                          <div className="stat-value">
                            {
                              data[index].visualize.records[displayIndex]
                                .ball_round
                            }
                          </div>
                          {/* <div className="stat-desc"></div> */}
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-col gap-4 bg-secondary p-4 px-6 rounded-b-2xl">
                      <h2 className="font-bold text-xl">Operation</h2>
                      <div className="grid grid-cols-3 gap-8">
                        <button
                          className="btn btn-primary"
                          onClick={() =>
                            timer == null ? startTimer() : clearTimer()
                          }
                        >
                          {timer == null ? "Auto" : "Stop"}
                        </button>
                        <button
                          className="btn btn-primary"
                          onClick={() => prevBall()}
                          disabled={timer !== null}
                        >
                          Prev
                        </button>
                        <button
                          className="btn btn-primary"
                          onClick={() => nextBall()}
                          disabled={timer !== null}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  ""
                )}
              </div>
              {data.length ? (
                <Court
                  data={data[index].visualize.records}
                  index={displayIndex}
                />
              ) : (
                ""
              )}
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
                  <h2 className="card-title">Options</h2>
                  <div className="flex gap-2">
                    <label className="label justify-normal gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={opponentType}
                        onChange={(e) => setOpponentType(e.target.checked)}
                        className="checkbox checkbox-primary"
                      />
                      <span className="label-text">Opponent Type</span>
                    </label>
                    <label className="label justify-normal gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={playerLocationArea}
                        onChange={(e) =>
                          setPlayerLocationArea(e.target.checked)
                        }
                        className="checkbox checkbox-primary"
                      />
                      <span className="label-text">Player Location Area</span>
                    </label>
                    <label className="label justify-normal gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={opponentLocationArea}
                        onChange={(e) =>
                          setOpponentLocationArea(e.target.checked)
                        }
                        className="checkbox checkbox-primary"
                      />
                      <span className="label-text">Opponent Location Area</span>
                    </label>
                    <label className="label justify-normal gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={hitArea}
                        onChange={(e) => setHitArea(e.target.checked)}
                        className="checkbox checkbox-primary"
                      />
                      <span className="label-text">Hit Area</span>
                    </label>
                  </div>
                </div>
              </div>

              <button className="btn btn-primary mx-4" type="submit">
                Analysis
              </button>
            </form>
          )}
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

export default Visualize;
