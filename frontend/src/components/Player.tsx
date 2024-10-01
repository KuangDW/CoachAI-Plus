import { useEffect, useRef } from "react";
import anime from "animejs/lib/anime.es.js";

const Player = ({
  x,
  y,
  variant,
}: {
  x: number;
  y: number;
  variant: "A" | "B";
}) => {
  const PlayerRef = useRef(null);

  useEffect(() => moveTo(x, y), [x, y]);

  const moveTo = (x: number, y: number) => {
    anime({
      targets: PlayerRef.current,
      translateX: x,
      translateY: y,
      duration: 1000,
      easing: "easeInOutQuad",
    });
  };

  return (
    <div
      ref={PlayerRef}
      className={
        "w-6 h-6 rounded-full absolute left-[165.5px] top-[480px] text-center select-none " +
        (variant === "A" ? "bg-red-500" : "bg-blue-500")
      }
    >
      {variant}
    </div>
  );
};

export default Player;
