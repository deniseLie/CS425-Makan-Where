import { wordDiff } from './diff.js'
import { SLOTS } from './SlotBoard.jsx'

const CRITIQUE_LABEL = { too_far: 'too far', too_expensive: 'too expensive', not_interested: 'not interested' }
export const SOURCE_LABEL = { llm: 'fine-tuned LLM', template: 'template', guardrail: 'guardrail → template' }

// One card per stage. `key` matches debug.timings_ms and the stream's stage events.
export const STAGES = [
  { key: 'raw', title: 'You said' },
  { key: 'normalize', title: 'Normalizer', n: 1 },
  { key: 'style', title: 'Style', n: 2 },
  { key: 'track', title: 'Tracker', n: 3 },
  { key: 'policy', title: 'Policy', n: 4 },
  { key: 'generate', title: 'Generator', n: 5 },
]

export function seconds(ms) {
  if (ms == null) return null
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`
}

function show(value) {
  if (value === 'UNKNOWN') return '?'
  if (value === 'ANY') return 'any'
  return value
}

/** Slots the tracker changed this turn, with what the user said when translate() changed it
 *  ("Bugis" -> "downtown core"; a change of case alone doesn't count). */
export function slotChanges(debug, prevState) {
  const rawSlots = debug.tracker_raw?.slots ?? {}
  return SLOTS.filter((s) => debug.state.slots[s] !== (prevState?.slots[s] ?? 'UNKNOWN')).map((s) => {
    const said = rawSlots[s]
    const value = debug.state.slots[s]
    const translated = said && said.toLowerCase() !== value.toLowerCase() && !['UNKNOWN', 'ANY'].includes(said)
    return { slot: s, said: translated ? said : null, value }
  })
}

/** What each finished stage did, and whether it changed anything worth pointing at. Works on a
 *  partial debug while a turn streams in: a stage appears once its output has arrived. */
function describe(debug, prevDebug) {
  const out = { raw: { changed: false, body: <q>{debug.raw}</q> } }
  const prevState = prevDebug?.state

  if (debug.normalized != null) {
    const diff = wordDiff(debug.raw, debug.normalized)
    const rewrote = diff.some((d) => d.op !== 'same')
    out.normalize = {
      changed: rewrote,
      note: rewrote ? 'rewrote' : 'no change',
      body: <span className="wdiff">{diff.map((d, i) => <span key={i} className={d.op}>{d.text}</span>)}</span>,
    }
  }

  if (debug.style != null) {
    out.style = {
      changed: !prevDebug || prevDebug.style !== debug.style,
      body: debug.style === 'singlish' ? <><b>Singlish</b> → reply in Singlish</> : <><b>English</b> → reply in English</>,
    }
  }

  if (debug.state != null) {
    const changes = slotChanges(debug, prevState)
    const critique = debug.tracker_output?.critique
    out.track = {
      changed: changes.length > 0 || !!critique,
      note: changes.length ? `${changes.length} slot${changes.length === 1 ? '' : 's'}` : null,
      body: changes.length || critique ? (
        <ul className="changes">
          {changes.map((c) => (
            <li key={c.slot}>
              <span className="k">{c.slot}</span>
              {c.said && <><span className="said">{c.said}</span><span className="arrow">→</span></>}
              <b>{show(c.value)}</b>
            </li>
          ))}
          {critique && <li><span className="k">critique</span><b className="crit">{CRITIQUE_LABEL[critique] ?? critique}</b></li>}
        </ul>
      ) : <span className="muted">nothing new this turn</span>,
    }
  }

  if (debug.action != null) {
    const { action, state } = debug
    const radiusFrom = prevState?.radius_km ?? 2
    const n = action.places.length
    out.policy = {
      changed: true,
      note: `${debug.n_asked} asked`,
      body: (
        <>
          {action.kind === 'ask' && <>ask about <b>{action.ask_slot}</b></>}
          {action.kind === 'fallback_recommend' && <>recommend <b>{n}</b>, relaxing <b>{action.relaxed_slot}</b></>}
          {action.kind === 'recommend' && <>recommend <b>{n}</b> place{n === 1 ? '' : 's'}</>}
          {state && radiusFrom !== state.radius_km && <div className="sub">radius {radiusFrom} → <b>{state.radius_km} km</b></div>}
        </>
      ),
    }
  }

  if (debug.reply_source != null) {
    const source = debug.reply_source
    out.generate = {
      changed: source !== 'llm',
      warn: source === 'guardrail',
      note: SOURCE_LABEL[source] ?? source,
      body: debug.reply ? <span className="reply-line">{debug.reply.split('\n')[0]}</span> : <span className="muted">reply ready</span>,
    }
  }
  return out
}

/** The architecture slide, live: one card per pipeline stage. While a turn runs, each card fills in
 *  with the backend's real output the moment its stage finishes, and the next one pulses. */
export default function PipelineFlow({ debug, prevDebug, running }) {
  if (!debug) return <p className="muted">Send a message to watch it flow through the pipeline.</p>
  const info = describe(debug, prevDebug)
  const timings = debug.timings_ms ?? {}
  const current = running ? STAGES.find((s) => !info[s.key])?.key : null

  return (
    <ol className="flow">
      {STAGES.map((s, i) => {
        const d = info[s.key]
        const status = d ? (d.changed ? 'lit' : 'quiet') : s.key === current ? 'running' : 'waiting'
        return (
          <li key={s.key} className={`stage ${status}${d?.warn ? ' warn' : ''}`} style={{ '--i': running ? 0 : i }}>
            <div className="stage-head">
              <span className="stage-num" aria-hidden="true">{s.n ?? '›'}</span>
              <span className="stage-title">{s.title}</span>
              <span className="stage-right">
                {d?.note && <span className="stage-note">{d.note}</span>}
                {timings[s.key] != null && <span className="stage-time">{seconds(timings[s.key])}</span>}
                {status === 'running' && <span className="stage-spin" aria-label="running" />}
              </span>
            </div>
            <div className="stage-body">
              {d ? d.body : <span className="muted">{status === 'running' ? 'working…' : 'waiting'}</span>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
