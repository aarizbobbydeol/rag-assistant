import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The FastAPI service in development. Every backend path is proxied through the
 * dev server on the same origin, which is why the app never needs a CORS story
 * locally and why `VITE_API_BASE_URL` can stay empty.
 */
const API_TARGET = 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/metrics': { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
