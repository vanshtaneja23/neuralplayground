import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served from https://vanshtaneja23.github.io/neuralplayground/, so asset URLs
// must be prefixed with the repo name rather than resolving against the domain root.
export default defineConfig({
  plugins: [react()],
  base: "/neuralplayground/",
});
