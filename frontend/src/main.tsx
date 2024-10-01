import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import "./index.css";

import App from "./App";
import Shot from "./routes/Shot";
import Tactic from "./routes/Tactic";
import Visualization from "./routes/Visualization";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        path: "/",
        element: <Shot />,
      },
      {
        path: "/tactic",
        element: <Tactic />,
      },
      {
        path: "/visualization",
        element: <Visualization />,
      },
    ],
  },
]);

// Fixed document hight, let component have full control to hight
const documentHeight = () => {
  const doc = document.documentElement;
  doc.style.setProperty("--doc-height", `${window.innerHeight}px`);
};
window.addEventListener("resize", documentHeight);
documentHeight();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
