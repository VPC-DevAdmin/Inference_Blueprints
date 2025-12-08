import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Backend URL for the dev proxy. Keep this pointed at the container DNS name.
const backendTarget =
  process.env.BACKEND_PROXY_URL ||
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
