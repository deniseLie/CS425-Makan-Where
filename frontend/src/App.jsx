import { Fragment, useEffect, useRef, useState } from 'react'
import { createSession, health, resetSession, sendMessage } from './api.js'
import PinnedList from './PinnedList.jsx'
import PlaceCards from './PlaceCards.jsx'
import RealMap from './RealMap.jsx'

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
  const [hovered, setHovered] = useState(null) // place_id under the mouse, on a card or a pin
  const [selectedCard, setSelectedCard] = useState(null) // place_id of the clicked card
  const [focus, setFocus] = useState(null) // [lat, lon] the map should fly to
  // places the user pinned, kept across reloads; storage can be blocked, so never rely on it
  const [pinned, setPinned] = useState(() => {
    try { return JSON.parse(localStorage.getItem('makan.pins')) || {} } catch { return {} }
  })
  useEffect(() => {
    try { localStorage.setItem('makan.pins', JSON.stringify(pinned)) } catch { /* private window */ }
  }, [pinned])
  const focusOn = (p) => p.lat != null && setFocus([p.lat, p.lon])
  function selectCard(p) { setSelectedCard(p.place_id); focusOn(p) }
  function togglePin(p) {
    setPinned((cur) => {
      if (cur[p.place_id]) { const { [p.place_id]: _, ...rest } = cur; return rest }
      return { ...cur, [p.place_id]: p }
    })
    if (!pinned[p.place_id]) focusOn(p)
  }
  function unpin(id) { setPinned(({ [id]: _, ...rest }) => rest) }
  const endRef = useRef(null)

  useEffect(() => {
    health().then((h) => setComponents(h.components)).catch(() => setError('Backend not reachable. Is uvicorn running on port 8000?'))
    // keep an existing session: a hot reload re-runs this effect mid-conversation
    createSession().then((id) => setSessionId((cur) => cur ?? id)).catch(() => {})
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
      setMessages((m) => [...m, { role: 'bot', text: res.reply, places: res.places ?? [], search: res.search ?? null }])
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

  // In the chat the newest recommendation is live and older ones are superseded. The map shows the
  // newest picks; the set before those supplies ghost pins (picks that dropped out).
  const pickTurns = messages.flatMap((m, i) => (m.places?.length ? [i] : []))
  const latest = pickTurns.at(-1)
  const livePlaces = latest != null ? messages[latest].places : []
  const liveIds = new Set(livePlaces.map((p) => p.place_id))
  const ghosts = pickTurns.length > 1 ? messages[pickTurns.at(-2)].places.filter((p) => !liveIds.has(p.place_id)) : []
  const search = messages.findLast((m) => m.role === 'bot')?.search ?? null

  return (
    <div className="app">
      <header>
        <h1 className="brand">
          <img src="/logo.svg" alt="" width="40" height="40" />
          Makan <span className="accent">Where?</span>
        </h1>
        {/* one quiet status dot instead of a badge per model; the model names are in the tooltip */}
        <div className={`status ${components ? 'ready' : error.startsWith('Backend') ? 'offline' : 'connecting'}`}
          title={components ? ['normalizer', 'style', 'tracker', 'generator']
            .map((c) => `${c}: ${components[c]?.model ?? '?'}`).join('\n') : undefined}>
          <i aria-hidden="true" />{components ? 'Ready' : error.startsWith('Backend') ? 'Offline' : 'Connecting…'}
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
            {messages.map((m, i) => (
              <Fragment key={i}>
                <div className={`bubble ${m.role}`}>{m.text}</div>
                {m.places?.length > 0 && (
                  <PlaceCards places={m.places} superseded={i !== latest} hovered={hovered} onHover={setHovered}
                    selectedCard={selectedCard} onSelect={selectCard} pinned={pinned} onTogglePin={togglePin} />
                )}
              </Fragment>
            ))}
            {busy && <div className="bubble bot typing" role="status" aria-label="Working on a reply"><i /><i /><i /></div>}
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
          <section>
            <h2>Pipeline state</h2>
            <Debug debug={debug} />
          </section>
          <section>
            <h2>Search area</h2>
            <RealMap search={search} places={livePlaces} ghosts={ghosts} hovered={hovered} onHover={setHovered}
              pinned={pinned} onUnpin={unpin} focus={focus} />
            <PinnedList pinned={pinned} center={search?.center} onFocus={focusOn} onUnpin={unpin} onClear={() => setPinned({})} />
          </section>
        </aside>
      </main>
    </div>
  )
}
