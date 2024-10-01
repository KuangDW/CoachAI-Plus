export const TacticDisplay = ({ data }: { data: TacticResponse }) => {
  const getImgUrl = (id: string) => {
    return `${
      import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""
    }/statics/${id}.png`;
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2">
        {Object.keys(data).map((key, i) =>
          Object.keys(data).map((key2, j) => {
            if (i < j)
              return <img src={getImgUrl(data[key][key2].histogram)} />;
            else "";
          })
        )}
        {Object.keys(data).map((key, i) =>
          Object.keys(data).map((key2, j) => {
            if (i < j)
              return <img src={getImgUrl(data[key][key2].coord_diagram)} />;
            else "";
          })
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {Object.keys(data).map((key) => (
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title">{key}</h2>
              <img src={getImgUrl(data[key].pie_chart)} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TacticDisplay;
