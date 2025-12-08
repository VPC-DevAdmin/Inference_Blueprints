import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Backend URL for dev/proxy. Prefer VITE_API_URL to match frontend config elsewhere.
const backendTarget =
  process.env.VITE_API_URL ||
  process.env.VITE_API_BASE_URL ||
  'http://rag-backend:5001';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
  },
});
