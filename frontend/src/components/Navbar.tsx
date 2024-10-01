import { NavLink } from "react-router-dom";

export const Navbar = () => {
  const routes = [
    {
      name: "Shot",
      path: "/",
    },
    {
      name: "Tactic",
      path: "/tactic",
    },
    {
      name: "Visualization",
      path: "/visualization",
    },
  ];

  return (
    <div className="navbar bg-base-100 shadow-xl shrink-0">
      <div className="flex-1">
        <h2 className="text-xl font-bold px-4">CoachAI+</h2>
        {routes.map((route) => (
          <NavLink
            to={route.path}
            key={route.path}
            className={({ isActive }) =>
              isActive ? "btn btn-primary" : "btn btn-ghost"
            }
          >
            {route.name}
          </NavLink>
        ))}
      </div>
    </div>
  );
};

export default Navbar;
