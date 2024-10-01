import { useEffect, useRef } from "react";
import anime from "animejs/lib/anime.es.js";

const Shuttlecock = ({ x, y, missed = false }: { x: number; y: number, missed: boolean }) => {
  const shuttlecockRef = useRef(null);

  useEffect(() => moveTo(x, y), [x, y])

  // Move shuttlecock to specific position
  const moveTo = (x: number, y: number) => {
    anime({
      targets: shuttlecockRef.current,
      translateX: x,
      translateY: y,
      duration: 1000,
      easing: "easeInOutQuad",
    });
  };

  return (
    <div
      ref={shuttlecockRef}
      className={"w-3 h-3 rounded-full absolute left-[171.5px] top-[480px] " + (missed ? "bg-pink-500" : "bg-yellow-500")}
    />
  );
};

export default Shuttlecock;
