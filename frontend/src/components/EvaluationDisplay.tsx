import { useState } from "react";

const ShotInfluence = ({ ids }: { ids: string[] }) => {
  const getImgUrl = (id: string) => {
    return `${
      import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""
    }/statics/${id}.png`;
  };

  const [index, setIndex] = useState(0);

  return (
    <div className="flex flex-col gap-2">
      <img src={getImgUrl(ids[index])} />
      <div className="flex justify-between items-center">
        <button
          className="btn btn-primary"
          disabled={index <= 0}
          onClick={() => setIndex((i) => i - 1)}
        >
          Prev
        </button>
        <button
          className="btn btn-primary"
          disabled={index >= ids.length - 1}
          onClick={() => setIndex((i) => i + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export const EvaluationDisplay = ({ data }: { data: EvaluationResponse }) => {
  const getImgUrl = (id: string) => {
    return `${
      import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""
    }/statics/${id}.png`;
  };

  return (
    <>
      <div className="flex items-center justify-center">
        <img src={getImgUrl(data.energetic_cost)} className="max-h-96" />
      </div>
      <div className="grid grid-cols-2 gap-6">
        {Object.keys(data)
          .filter((k) => k != "energetic_cost")
          .map((key) => (
            <div className="card bg-base-100 shadow-xl">
              <div className="card-body">
                <h1 className="font-bold text-3xl">{key}</h1>
                <h2 className="card-title">Shot Type</h2>
                <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                  {data[key].shot_type.map((s) => (
                    <img key={s} src={getImgUrl(s)} />
                  ))}
                </div>
                <h2 className="card-title">Last Ball</h2>
                <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                  <img src={getImgUrl(data[key].last_ball["Smash"])} />
                  <img src={getImgUrl(data[key].last_ball["Push Shot"])} />
                  <img src={getImgUrl(data[key].last_ball["Lob"])} />
                  <img src={getImgUrl(data[key].last_ball["Smash Defence"])} />
                  <img src={getImgUrl(data[key].last_ball["Net Shot"])} />
                  <img src={getImgUrl(data[key].last_ball["Drop"])} />
                  <img src={getImgUrl(data[key].last_ball["Clear"])} />
                  <img src={getImgUrl(data[key].last_ball["Drive"])} />
                </div>
                <h2 className="card-title">Top Reasons</h2>
                <img src={getImgUrl(data[key].top_reasons.win)} />
                <img src={getImgUrl(data[key].top_reasons.loose)} />
                <h2 className="card-title">Shot Influence</h2>
                <ShotInfluence ids={data[key].shot_influence} />
              </div>
            </div>
          ))}
      </div>
    </>
  );
};

export default EvaluationDisplay;
