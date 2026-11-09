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

/** Send a message and stream the turn: `onEvent` gets {type: 'start' | 'stage' | 'done' | 'error'}
 *  as server-sent events arrive. Resolves with the final 'done' payload (same shape as sendMessage). */
export async function streamMessage(id, message, onEvent) {
  const res = await fetch(`${BASE}/api/sessions/${id}/messages/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}`)
    err.status = res.status
    throw err
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let done = null
  for (;;) {
    const { value, done: finished } = await reader.read()
    if (value) buffer += decoder.decode(value, { stream: true })
    let cut
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      const data = chunk.split('\n').filter((l) => l.startsWith('data: ')).map((l) => l.slice(6)).join('\n')
      if (!data) continue
      const event = JSON.parse(data)
      onEvent(event)
      if (event.type === 'done') done = event
      if (event.type === 'error') throw new Error(event.message)
    }
    if (finished) break
  }
  if (!done) throw new Error('the reply stream ended early')
  return done
}
