import players from "../assets/player.json";
import { useState } from "react";

export const AgentSelection = ({
  title = "Agent",
  agent = "CHOU Tien Chen",
  changeAgent,
  customAgent = false,
  changeCustomAgent,
}: {
  title: string;
  agent: string;
  changeAgent?: (m: string) => void;
  customAgent: boolean
  changeCustomAgent?: (m: boolean) => void;
}) => {

  const [value, setValue] = useState(customAgent ? "custom" : "default")

  const onChangeMode = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue((e.target as HTMLInputElement).value)
    if (changeCustomAgent) {
      changeCustomAgent(value === "custom");
    }
  };

  const onChangeAgent = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (changeAgent) {
      changeAgent((e.target as HTMLSelectElement).value);
    }
  };

  return (
    <div className="card bg-base-100 shadow-xl mx-4">
      <div className="card-body">
        <h2 className="card-title">{title}</h2>
        <div className="flex gap-2 items-center">
          <input
            type="radio"
            name={title + "Selection"}
            className="radio radio-primary"
            checked={value === "default"}
            value="default"
            onChange={onChangeMode}
          />
          <span>Default Agent</span>
          <select className="select select-bordered select-sm max-w-xs" value={agent} onChange={onChangeAgent} disabled={value === "custom"}>
            {Object.keys(players).map((player) => {
              return <option key={player}>{player}</option>;
            })}
          </select>
        </div>
        <div className="flex gap-2 items-center">
          <input
            type="radio"
            name={title + "Selection"}
            className="radio radio-primary"
            checked={value === "custom"}
            value="custom"
            onChange={onChangeMode}
          />
          <span>Other Agent</span>
          <input
            type="file"
            className="file-input file-input-bordered file-input-sm max-w-xs"
            disabled={value === "default"}
          />
        </div>
      </div>
    </div>
  );
};
