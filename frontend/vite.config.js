import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api goes to the FastAPI backend (uvicorn on 8000) during development.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': process.env.VITE_API_TARGET || 'http://127.0.0.1:8000' } },
})
