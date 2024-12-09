export const TacticDisplay = ({ data }: { data: TacticResponse }) => {
  const getImgUrl = (id: string) => {
    return `${
      import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""
    }/statics/${id}.png`;
  };

  const abbreviation = [
    { tactic: "Full_Court_Pressure", abbrev: "FCP" },
    { tactic: "Defensive_Counterattack", abbrev: "DC" },
    { tactic: "Four_Corner", abbrev: "FC" },
    { tactic: "Forehand_Lock", abbrev: "FhL" },
    { tactic: "Backhand_Lock", abbrev: "BhL" },
    { tactic: "FrontCourt_Lock", abbrev: "FcL" },
    { tactic: "BackCourt_Lock", abbrev: "BcL" },
    { tactic: "Four_Corners_Clear_Drop", abbrev: "FCCD" },
    { tactic: "No_tactic", abbrev: "No" },
  ];

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2">
        {data.players.map((key, i) =>
          data.players.map((key2, j) => {
            if (i < j)
              return <img src={getImgUrl(data[key][key2].histogram)} />;
            else "";
          })
        )}
        {data.players.map((key, i) =>
          data.players.map((key2, j) => {
            if (i < j)
              return <img src={getImgUrl(data[key][key2].coord_diagram)} />;
            else "";
          })
        )}
      </div>
      <div className="px-8">
        <table className="table table-zebra">
          <thead>
            <tr>
              <th>Tactic</th>
              <th>Abbreviation</th>
            </tr>
          </thead>
          <tbody>
            {abbreviation.map((v) => (
              <tr>
                <td>{v.tactic}</td>
                <td>{v.abbrev}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {data.players.map((key) => (
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
