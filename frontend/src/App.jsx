import { Fragment, useEffect, useRef, useState } from 'react'
import { createSession, health, resetSession, streamMessage } from './api.js'
import LiveLog, { logLine } from './LiveLog.jsx'
import PinnedList from './PinnedList.jsx'
import RealMap from './RealMap.jsx'
import PipelineFlow from './PipelineFlow.jsx'
import PlaceCards from './PlaceCards.jsx'
import SlotBoard from './SlotBoard.jsx'
import useTurnClock from './useTurnClock.js'

const SUGGESTIONS = ['I want eat something leh', 'korean lah, bugis side', 'must be halal hor', 'too far leh']

export default function App() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [components, setComponents] = useState(null)
  const [selected, setSelected] = useState(null) // index of a past bot message to inspect; null = latest
  const [live, setLive] = useState(null) // the turn streaming in: {debug: stages so far, log, failed}
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
  const elapsed = useTurnClock(busy)

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
    setSelected(null)
    setMessages((m) => [...m, { role: 'user', text: message }])
    // the pipeline view fills in from the stream: each stage's real output as it finishes
    const t0 = performance.now()
    const partial = { raw: message, timings_ms: {} }
    const log = []
    const prevState = messages.findLast((m) => m.debug)?.debug.state
    const onEvent = (ev) => {
      if (ev.type === 'stage') {
        const { type, stage, ms, ...output } = ev
        Object.assign(partial, output)
        partial.timings_ms = { ...partial.timings_ms, [stage]: ms }
      }
      const line = logLine(ev, performance.now() - t0, partial, prevState)
      if (line) log.push(line)
      setLive({ debug: { ...partial }, log: [...log] })
    }
    setLive({ debug: { ...partial }, log: [] })
    try {
      let id = sessionId || (await createSession())
      let res
      try {
        res = await streamMessage(id, message, onEvent)
      } catch (e) {
        if (e.status !== 404) throw e
        id = await createSession() // backend restarted: start a fresh session
        res = await streamMessage(id, message, onEvent)
      }
      setSessionId(id)
      setMessages((m) => [...m, { role: 'bot', text: res.reply, places: res.places ?? [], search: res.search ?? null, debug: res.debug, log }])
      setLive(null)
    } catch (e) {
      setError(`Could not get a reply (${e.message}).`)
      if (!log.some((l) => l.kind === 'error')) log.push({ t: performance.now() - t0, tag: 'error', text: e.message, kind: 'error' })
      setLive({ debug: { ...partial }, log: [...log], failed: true })
    } finally {
      setBusy(false)
    }
  }

  async function newChat() {
    if (sessionId) await resetSession(sessionId).catch(() => setSessionId(null))
    setMessages([])
    setSelected(null)
    setLive(null)
    setError('')
  }

  // The side panel shows one turn: the latest, or a past one the user clicked.
  const turns = messages.flatMap((m, i) => (m.debug ? [i] : []))
  const viewIdx = selected ?? turns.at(-1)
  const viewPos = turns.indexOf(viewIdx)
  const debug = viewIdx != null ? messages[viewIdx].debug : null
  const prevDebug = viewPos > 0 ? messages[turns[viewPos - 1]].debug : null
  const viewingPast = selected != null && selected !== turns.at(-1)

  // In the chat the newest recommendation is live and older ones are superseded. The map shows the
  // newest picks up to the viewed turn; the set before those supplies ghost pins (picks that dropped out).
  const pickTurns = messages.flatMap((m, i) => (m.places?.length ? [i] : []))
  const latest = pickTurns.at(-1)
  const mapTurns = pickTurns.filter((i) => viewIdx != null && i <= viewIdx)
  const livePlaces = mapTurns.length ? messages[mapTurns.at(-1)].places : []
  const liveIds = new Set(livePlaces.map((p) => p.place_id))
  const ghosts = mapTurns.length > 1 ? messages[mapTurns.at(-2)].places.filter((p) => !liveIds.has(p.place_id)) : []
  const search = viewIdx != null ? messages[viewIdx].search : null
  const running = busy && !viewingPast

  function inspect(userIdx) {
    const bot = userIdx + 1
    if (!messages[bot]?.debug) return
    setSelected(bot === turns.at(-1) ? null : bot)
  }

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
                {m.role === 'user' && messages[i + 1]?.debug ? (
                  <button type="button" className={`bubble user inspectable${viewingPast && viewIdx === i + 1 ? ' viewing' : ''}`}
                    onClick={() => inspect(i)} title="Show this turn in the pipeline">{m.text}</button>
                ) : (
                  <div className={`bubble ${m.role}`}>{m.text}</div>
                )}
                {m.places?.length > 0 && (
                  <PlaceCards places={m.places} superseded={i !== latest} hovered={hovered} onHover={setHovered}
                    selectedCard={selectedCard} onSelect={selectCard} pinned={pinned} onTogglePin={togglePin} />
                )}
              </Fragment>
            ))}
            {/* the chat just shows it's typing; the pipeline panel shows which steps are done */}
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

        <aside className="debug" aria-label="Pipeline">
          {viewingPast && (
            <div className="viewing-banner">
              Viewing turn {viewPos + 1} of {turns.length}
              <button type="button" className="secondary" onClick={() => setSelected(null)}>Back to latest</button>
            </div>
          )}
          <section className="flow-col">
            <h2>Pipeline{debug && !running ? ` · turn ${viewPos + 1}` : ''}</h2>
            {/* same key while streaming and after, so finished cards don't flash twice */}
            <PipelineFlow key={running ? messages.length : viewIdx} running={running}
              debug={running && live ? live.debug : debug} prevDebug={running ? messages.findLast((m) => m.debug)?.debug : prevDebug} />
            <h2 className="log-head">Pipeline log{running ? <><span className="rec" aria-hidden="true" /><span className="log-elapsed">{Math.floor(elapsed / 1000)} s</span></> : null}</h2>
            <LiveLog running={running} lines={live && !viewingPast ? live.log : debug ? messages[viewIdx].log : null} />
          </section>
          <div className="state-col">
            <section>
              <h2>Dialogue state</h2>
              {debug ? (
                <SlotBoard state={debug.state} prevState={prevDebug?.state} critique={debug.tracker_output?.critique}
                  turnIdx={`${viewIdx}`} />
              ) : <p className="muted">Slots fill in as you chat.</p>}
            </section>
            <section>
              <h2>Search area</h2>
              <RealMap search={search} places={livePlaces} ghosts={ghosts} hovered={hovered} onHover={setHovered}
                pinned={pinned} onUnpin={unpin} focus={focus} />
              <PinnedList pinned={pinned} center={search?.center} onFocus={focusOn} onUnpin={unpin} onClear={() => setPinned({})} />
            </section>
            {debug && (
              <details>
                <summary>Raw JSON</summary>
                <pre>{JSON.stringify(debug, null, 2)}</pre>
              </details>
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}
