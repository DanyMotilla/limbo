import { defineConfig } from "vite";
import reactPlugin from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [reactPlugin()],
  optimizeDeps: {
    include: ['@mui/material', '@emotion/react', '@emotion/styled'],
  },
  build: {
    outDir: "build",
  },
  server: {
    port: 4444,
  },
});
