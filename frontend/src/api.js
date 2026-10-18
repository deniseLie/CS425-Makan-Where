// Thin client for backend/src/api.py. VITE_API_BASE lets a deployed frontend call another host.
const BASE = import.meta.env.VITE_API_BASE || ''

async function call(path, options = {}) {
  const res = await fetch(`${BASE}/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}`)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const health = () => call('/health')
export const createSession = () => call('/sessions', { method: 'POST' }).then((r) => r.session_id)
export const sendMessage = (id, message) =>
  call(`/sessions/${id}/messages`, { method: 'POST', body: JSON.stringify({ message }) })
export const resetSession = (id) => call(`/sessions/${id}/reset`, { method: 'POST' })
