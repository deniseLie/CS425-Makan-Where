import { haversineKm } from './RealMap.jsx'

/** The places the user pinned: click a row to fly to it, ✕ to remove it. */
export default function PinnedList({ pinned, center, onFocus, onUnpin, onClear }) {
  const list = Object.values(pinned)
  if (!list.length) return null
  return (
    <div className="pinned">
      <div className="pinned-head">
        <span>Pinned ({list.length})</span>
        <button type="button" className="link" onClick={onClear}>Clear all</button>
      </div>
      <ul>
        {list.map((p) => {
          const km = center && p.lat != null ? haversineKm(center, [p.lat, p.lon]) : null
          return (
            <li key={p.place_id}>
              <button type="button" className="pinned-name" onClick={() => onFocus(p)}>{p.name}</button>
              {km != null && <span className="muted">{km < 1 ? `${Math.round(km * 1000 / 50) * 50} m` : `${km.toFixed(1)} km`}</span>}
              <button type="button" className="pinned-x" onClick={() => onUnpin(p.place_id)} aria-label={`Remove ${p.name}`}>✕</button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
