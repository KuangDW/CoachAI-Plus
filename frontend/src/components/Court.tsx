import Shuttlecock from "./Shuttlecock";
import Player from "./Player";
import CourtImg from "../assets/Court.png";

export const Court = ({ data, index }: {data: DisplayRallyData[], index: number}) => {

  const currentState = () => {
    return data[index];
  };

  return (
    <>
      <div className="block relative w-[355px] shrink-0 bg-[#198464]">
        <img src={CourtImg} className="absolute" />
        <Player
          x={currentState().player_location_x}
          y={currentState().player_location_y}
          variant="A"
        />
        <Player
          x={currentState().opponent_location_x}
          y={currentState().opponent_location_y}
          variant="B"
        />
        <Shuttlecock
          x={currentState().ball_location_x}
          y={currentState().ball_location_y}
          missed={false}
        />
      </div>
    </>
  );
};

export default Court;
