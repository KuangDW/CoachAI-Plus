export const VisualizeDisplay = ({ data }: { data: VisualizeResponse }) => {

  const getImgUrl = (id: string) => {
    return `${import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""}/statics/${id}.png`
  }

  return (
    <div className="grid grid-cols-2 gap-6">
      {data.players.map((key) => (
        <div className="card bg-base-100 shadow-xl" key={key}>
          <div className="card-body">
            <h2 className="card-title">{key}</h2>
            <img src={getImgUrl(data[key].ShotType)} />
            <img src={getImgUrl(data[key].top_win_states)} />
            <img src={getImgUrl(data[key].top_lose_states)} />
          </div>
        </div>
      ))}
    </div>
  );
};

export default VisualizeDisplay;
