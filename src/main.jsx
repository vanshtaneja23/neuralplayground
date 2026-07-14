import React from "react";
import { createRoot } from "react-dom/client";
import NeuralPlayground from "../neural-playground.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <NeuralPlayground />
  </React.StrictMode>
);
