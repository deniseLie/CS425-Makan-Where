import { useEffect, useRef } from 'react'
import { SOURCE_LABEL, seconds, slotChanges } from './PipelineFlow.jsx'

const NAME = { normalize: 'normalizer', style: 'style', track: 'tracker', policy: 'policy', generate: 'generator' }
const CRITIQUE = { too_far: 'too far', too_expensive: 'too expensive', not_interested: 'not interested' }

/** One log line for one stream event. `partial` is the turn so far (raw text plus earlier stages)
 *  and `prevState` the dialogue state before this turn. */
export function logLine(event, t, partial, prevState) {
  const line = (tag, text, kind = '') => ({ t, tag, text, kind })
  switch (event.type) {
    case 'start':
      return line('send', `"${partial.raw}"`, 'muted')
    case 'error':
      return line('error', event.message, 'error')
    case 'done':
      return line('done', `reply ready (${seconds(Math.round(t))})`, 'ok')
    case 'stage': {
      const took = `(${seconds(event.ms)})`
      const tag = NAME[event.stage] ?? event.stage
      if (event.stage === 'normalize') {
        if (event.skipped) return line(tag, `skipped: English ${took}`, 'muted')
        const same = event.normalized.trim().toLowerCase() === partial.raw.trim().toLowerCase()
        return line(tag, same ? `unchanged ${took}` : `"${event.normalized}" ${took}`)
      }
      if (event.stage === 'style') return line(tag, `${event.style} ${took}`)
      if (event.stage === 'track') {
        const changes = slotChanges(event, prevState)
          .map((c) => `${c.slot}=${c.value}${c.said ? ` (said "${c.said}")` : ''}`)
        const critique = event.tracker_output?.critique
        if (critique) changes.push(`critique=${CRITIQUE[critique] ?? critique}`)
        return line(tag, `${changes.length ? changes.join(', ') : 'no new slots'} ${took}`)
      }
      if (event.stage === 'policy') {
        const a = event.action
        const radius = partial.state && prevState && partial.state.radius_km !== prevState.radius_km
          ? `, radius ${prevState.radius_km} → ${partial.state.radius_km} km` : ''
        if (a.kind === 'ask') return line(tag, `ask → ${a.ask_slot}${radius} ${took}`)
        const names = a.places.map((p) => p.name).join(', ')
        const what = a.kind === 'fallback_recommend' ? `relax ${a.relaxed_slot}, recommend` : 'recommend'
        return line(tag, `${what} ${a.places.length}${names ? `: ${names}` : ''}${radius} ${took}`)
      }
      if (event.stage === 'generate') {
        return line(tag, `${SOURCE_LABEL[event.reply_source] ?? event.reply_source} wrote the reply ${took}`,
          event.reply_source === 'guardrail' ? 'error' : '')
      }
      return line(tag, took)
    }
    default:
      return null
  }
}

/** Terminal-style log of one turn: what each stage returned and when. */
export default function LiveLog({ lines, running }) {
  const box = useRef(null)
  // scroll inside the log only (scrollIntoView would also scroll the whole side panel)
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight }, [lines?.length])
  if (!lines?.length) return <p className="muted">Each stage's output shows up here as it arrives.</p>
  return (
    <div className="log" role="log" aria-live="polite" ref={box}>
      {lines.map((l, i) => (
        <div key={i} className={`log-line ${l.kind}`}>
          <span className="log-t">{(l.t / 1000).toFixed(1).padStart(5, ' ')}s</span>
          <span className="log-tag">{l.tag}</span>
          <span className="log-text">{l.text}</span>
        </div>
      ))}
      {running && <div className="log-line muted"><span className="log-t" /><span className="log-cursor" /></div>}
    </div>
  )
}
