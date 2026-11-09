import { useRef } from 'react'

// Offline sketch map: no tiles, so it works on any projector without Wi-Fi.
const W = 320, H = 230, CX = W / 2, CY = H / 2, HALF = H / 2 - 18

function project(lat, lon, [lat0, lon0], pxPerKm) {
  const x = CX + (lon - lon0) * Math.cos((lat0 * Math.PI) / 180) * 111.32 * pxPerKm
  const y = CY - (lat - lat0) * 110.57 * pxPerKm
  return [Math.min(Math.max(x, 10), W - 10), Math.min(Math.max(y, 10), H - 10)]
}

/** Search centre, radius ring and numbered pins matching the cards. Picks dropped since the
 *  last recommendation stay as grey ghosts, so "too far leh" visibly pulls the search in. */
export default function MiniMap({ search, places, ghosts, hovered, onHover }) {
  const zoom = useRef({ key: null, km: 0 })
  const located = places.filter((p) => p.lat != null)
  const ghostPins = ghosts.filter((p) => p.lat != null)
  const all = [...located, ...ghostPins]
  if (!search?.center && all.length === 0) return <p className="muted">Name an area to see where I'm searching.</p>

  const centre = search?.center ?? [all.reduce((s, p) => s + p.lat, 0) / all.length, all.reduce((s, p) => s + p.lon, 0) / all.length]
  const far = Math.max(0, ...all.map((p) => p.distance_km ?? 0))
  // fit the radius, then only zoom out while the centre stays put: when the radius shrinks the
  // scale holds and the ring visibly gets smaller
  const key = centre.map((v) => v.toFixed(4)).join(',')
  const fit = Math.max(0.8, (search?.radius_km ?? 0) * 1.12, far * 1.15)
  if (zoom.current.key !== key) zoom.current = { key, km: 0 }
  zoom.current.km = Math.max(zoom.current.km, fit)
  const viewKm = zoom.current.km
  const pxPerKm = HALF / viewKm
  const ringR = search?.center ? search.radius_km * pxPerKm : 0
  const gridKm = viewKm > 6 ? 2 : 1

  return (
    <svg className="minimap" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Searching within ${search?.radius_km} km of ${search?.label ?? 'the chosen area'}, ${located.length} picks`}>
      <rect x="0" y="0" width={W} height={H} rx="10" className="map-bg" />
      {Array.from({ length: Math.floor(viewKm / gridKm) + 1 }, (_, i) => (i + 1) * gridKm).map((km) => (
        <circle key={km} cx={CX} cy={CY} r={km * pxPerKm} className="map-grid" />
      ))}
      {ringR > 0 && <circle cx={CX} cy={CY} r={ringR} style={{ r: ringR }} className="map-ring" />}
      {search?.center && (
        <g className="map-centre">
          <circle cx={CX} cy={CY} r="4" />
        </g>
      )}
      {ghostPins.map((p) => {
        const [x, y] = project(p.lat, p.lon, centre, pxPerKm)
        return (
          <g key={`ghost-${p.place_id}`} transform={`translate(${x} ${y})`} className="pin ghost">
            <title>{p.name} (dropped)</title>
            <circle r="7" />
          </g>
        )
      })}
      {located.map((p) => {
        const [x, y] = project(p.lat, p.lon, centre, pxPerKm)
        const i = places.indexOf(p)
        return (
          <g key={p.place_id} transform={`translate(${x} ${y})`} onMouseEnter={() => onHover(p.place_id)} onMouseLeave={() => onHover(null)}>
            <g className={`pin${hovered === p.place_id ? ' hot' : ''}`} style={{ '--i': i }}>
              <title>{p.name}</title>
              <circle r="10" />
              <text textAnchor="middle" dy="4">{i + 1}</text>
            </g>
          </g>
        )
      })}
      <g className="map-scale" transform={`translate(12 ${H - 12})`}>
        <line x1="0" x2={pxPerKm} y1="0" y2="0" />
        <text x={pxPerKm + 6} y="4">1 km</text>
      </g>
      {search?.center && <text className="map-radius" x={W - 12} y="20" textAnchor="end">radius {search.radius_km} km</text>}
      {search?.label && <text className="map-label" x="12" y="20">● around {search.label}</text>}
    </svg>
  )
}
