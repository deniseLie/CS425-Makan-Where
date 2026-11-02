import { useEffect, useRef, useState } from 'react'
import { createSession, health, resetSession, sendMessage } from './api.js'

const SUGGESTIONS = ['I want eat something leh', 'korean lah, bugis side', 'must be halal hor', 'too far leh']
const SLOTS = ['cuisine', 'area', 'budget', 'dietary', 'setting', 'time']

function Debug({ debug }) {
  if (!debug) return <p className="muted">Send a message to see the pipeline state.</p>
  const { action, state } = debug
  return (
    <>
      <dl className="kv">
        <dt>style</dt><dd>{debug.style}</dd>
        <dt>normalized</dt><dd>{debug.normalized}</dd>
        <dt>action</dt><dd>{action.kind}{action.ask_slot ? ` → ${action.ask_slot}` : ''}{action.relaxed_slot ? ` (relaxed ${action.relaxed_slot})` : ''}</dd>
        <dt>questions asked</dt><dd>{debug.n_asked}</dd>
        <dt>radius</dt><dd>{state.radius_km} km</dd>
        {debug.guardrail_triggered && <><dt>guardrail</dt><dd className="warn">replaced reply</dd></>}
      </dl>
      <table className="slots">
        <tbody>
          {SLOTS.map((s) => (
            <tr key={s}><th>{s}</th><td className={`slot-${state.slots[s]}`}>{state.slots[s]}</td></tr>
          ))}
        </tbody>
      </table>
      <details>
        <summary>Raw JSON</summary>
        <pre>{JSON.stringify(debug, null, 2)}</pre>
      </details>
    </>
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [debug, setDebug] = useState(null)
  const [components, setComponents] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    health().then((h) => setComponents(h.components)).catch(() => setError('Backend not reachable. Is uvicorn running on port 8000?'))
    createSession().then(setSessionId).catch(() => {})
  }, [])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function send(text) {
    const message = text.trim()
    if (!message || busy) return
    setBusy(true)
    setError('')
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: message }])
    try {
      let id = sessionId || (await createSession())
      let res
      try {
        res = await sendMessage(id, message)
      } catch (e) {
        if (e.status !== 404) throw e
        id = await createSession() // backend restarted: start a fresh session
        res = await sendMessage(id, message)
      }
      setSessionId(id)
      setMessages((m) => [...m, { role: 'bot', text: res.reply }])
      setDebug(res.debug)
    } catch (e) {
      setError(`Could not get a reply (${e.message}).`)
    } finally {
      setBusy(false)
    }
  }

  async function newChat() {
    if (sessionId) await resetSession(sessionId).catch(() => setSessionId(null))
    setMessages([])
    setDebug(null)
    setError('')
  }

  return (
    <div className="app">
      <header>
        <h1 className="brand">
          <img src="/logo.svg" alt="" width="40" height="40" />
          Makan <span className="accent">Where?</span>
        </h1>
        <div className="badges">
          {components && ['normalizer', 'style', 'tracker', 'generator'].map((c) => (
            <span key={c} className={`badge ${components[c].impl}`} title={components[c].model || 'local stub'}>
              {c}: {components[c].impl}
            </span>
          ))}
        </div>
      </header>

      <main>
        <section className="chat" aria-label="Conversation">
          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="empty">
                <p>Tell me what you feel like eating, Singlish also can.</p>
                <div className="chips">
                  {SUGGESTIONS.map((s) => <button key={s} onClick={() => setInput(s)}>{s}</button>)}
                </div>
              </div>
            )}
            {messages.map((m, i) => <div key={i} className={`bubble ${m.role}`}>{m.text}</div>)}
            {busy && <div className="bubble bot typing">…</div>}
            <div ref={endRef} />
          </div>
          {error && <p className="error" role="alert">{error}</p>}
          <form onSubmit={(e) => { e.preventDefault(); send(input) }}>
            <input value={input} onChange={(e) => setInput(e.target.value)} maxLength={1000}
              placeholder="e.g. Bugis side got halal food or not?" aria-label="Message" autoFocus />
            <button type="submit" disabled={busy || !input.trim()}>Send</button>
            <button type="button" className="secondary" onClick={newChat} disabled={busy}>New chat</button>
          </form>
        </section>

        <aside className="debug" aria-label="Pipeline debug">
          <h2>Pipeline state</h2>
          <Debug debug={debug} />
        </aside>
      </main>
    </div>
  )
}
