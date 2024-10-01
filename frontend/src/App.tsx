import { Outlet } from "react-router-dom";
import Navbar from "./components/Navbar";

const App = () => {
  return (
    <>
      <Navbar />
      <div className="grow overflow-auto">
        <Outlet />
      </div>
    </>
  );
};

export default App;
