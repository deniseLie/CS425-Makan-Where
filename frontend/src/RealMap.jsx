import L from 'leaflet'
import { useEffect, useRef, useState } from 'react'
import { Circle, CircleMarker, MapContainer, Marker, Popup, TileLayer, Tooltip, useMap } from 'react-leaflet'
import MiniMap from './MiniMap.jsx'

// OpenStreetMap's standard tiles: no key (CARTO's now watermarks keyless use). Dark mode darkens
// them with a CSS filter on the tile pane instead of a separate dark tile set.
const TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
const ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
const SINGAPORE = [1.3521, 103.8198]
const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

const icon = (cls, html) => L.divIcon({ className: '', html: `<div class="${cls}">${html}</div>`, iconSize: [26, 26], iconAnchor: [13, 13] })
const savedIcon = L.divIcon({ className: '', html: '<div class="saved-pin"></div>', iconSize: [24, 24], iconAnchor: [12, 24], popupAnchor: [0, -24] })

function useDark() {
  const q = window.matchMedia?.('(prefers-color-scheme: dark)')
  const [dark, setDark] = useState(!!q?.matches)
  useEffect(() => {
    if (!q) return
    const on = (e) => setDark(e.matches)
    q.addEventListener('change', on)
    return () => q.removeEventListener('change', on)
  }, [])
  return dark
}

export function haversineKm([lat1, lon1], [lat2, lon2]) {
  const r = (d) => (d * Math.PI) / 180
  const h = Math.sin(r(lat2 - lat1) / 2) ** 2 + Math.cos(r(lat1)) * Math.cos(r(lat2)) * Math.sin(r(lon2 - lon1) / 2) ** 2
  return 6371 * 2 * Math.asin(Math.sqrt(h))
}

/** Ring radius that eases from its old value to the new one, so "too far leh" visibly pulls it in. */
function useEasedRadius(target) {
  const [r, setR] = useState(target)
  const from = useRef(target)
  useEffect(() => {
    if (reducedMotion() || from.current === target) { setR(target); from.current = target; return }
    const start = performance.now(), a = from.current
    let id
    const step = (t) => {
      const k = Math.min(1, (t - start) / 800), e = 1 - (1 - k) ** 3
      setR(a + (target - a) * e)
      if (k < 1) id = requestAnimationFrame(step); else from.current = target
    }
    id = requestAnimationFrame(step)
    return () => cancelAnimationFrame(id)
  }, [target])
  return r
}

/** Fit the view when the search centre changes; when only the radius shrinks, hold the zoom. */
function Camera({ center, radiusKm, points, focus }) {
  const map = useMap()
  const key = center ? center.map((v) => v.toFixed(4)).join(',') : 'none'
  const fitted = useRef({ key: null, km: 0 })
  useEffect(() => {
    const pts = [...points]
    let bounds
    if (center) bounds = L.latLng(center).toBounds(radiusKm * 2000)
    for (const p of pts) bounds = bounds ? bounds.extend(p) : L.latLngBounds([p, p])
    if (!bounds) return
    const need = Math.max(radiusKm, ...pts.map((p) => (center ? haversineKm(center, p) : 0)))
    if (fitted.current.key === key && need <= fitted.current.km) return // radius shrank: keep zoom
    fitted.current = { key, km: need }
    map.fitBounds(bounds, { padding: [20, 20], maxZoom: 17, animate: !reducedMotion() })
  }, [key, radiusKm, points.length])
  useEffect(() => {
    if (!focus) return
    reducedMotion() ? map.setView(focus, Math.max(map.getZoom(), 16)) : map.flyTo(focus, Math.max(map.getZoom(), 16), { duration: 0.6 })
  }, [focus?.[0], focus?.[1]])
  useEffect(() => {
    const ro = new ResizeObserver(() => map.invalidateSize())
    ro.observe(map.getContainer())
    return () => ro.disconnect()
  }, [])
  return null
}

/** Count tile errors; a burst of them means no internet, so fall back to the offline sketch. */
function TileWatch({ onDead }) {
  const map = useMap()
  useEffect(() => {
    let errors = []
    const on = () => {
      const now = Date.now()
      errors = [...errors.filter((t) => now - t < 5000), now]
      if (errors.length >= 3) onDead()
    }
    map.on('tileerror', on)
    return () => map.off('tileerror', on)
  }, [])
  return null
}

function money(p) {
  const { price_min: lo, price_max: hi } = p
  if (lo == null && hi == null) return null
  return lo != null && hi != null && hi !== lo ? `$${lo}–${hi}` : `~$${hi ?? lo}`
}

/** Street map: search centre, radius ring, numbered picks, grey ghosts for dropped picks, and the
 *  places the user pinned (each with a popup to remove it). Falls back to MiniMap when offline. */
export default function RealMap({ search, places, ghosts, hovered, onHover, pinned, onUnpin, focus }) {
  const dark = useDark()
  const forceSketch = new URLSearchParams(location.search).get('map') === 'sketch'
  const [dead, setDead] = useState(forceSketch)
  const [big, setBig] = useState(false)
  const radius = useEasedRadius((search?.radius_km ?? 0) * 1000)

  const located = places.filter((p) => p.lat != null)
  const ghostPins = ghosts.filter((p) => p.lat != null)
  const saved = Object.values(pinned).filter((p) => p.lat != null)
  const pickIds = new Set(located.map((p) => p.place_id))
  const center = search?.center ?? null

  if (dead) return <MiniMap search={search} places={places} ghosts={ghosts} hovered={hovered} onHover={onHover} />
  if (!center && located.length + saved.length === 0) return <p className="muted">Name an area to see where I'm searching.</p>

  const points = [...located, ...ghostPins, ...saved].map((p) => [p.lat, p.lon])
  return (
    <div className={`realmap${big ? ' big' : ''}${dark ? ' dark' : ''}`}>
      <MapContainer center={center ?? points[0] ?? SINGAPORE} zoom={15} scrollWheelZoom={false} className="realmap-canvas"
        whenReady={(e) => { const m = e.target; m.on('click', () => m.scrollWheelZoom.enable()); m.on('mouseout', () => m.scrollWheelZoom.disable()) }}>
        <TileLayer url={TILES} attribution={ATTRIBUTION} maxZoom={19} />
        <TileWatch onDead={() => setDead(true)} />
        <Camera center={center} radiusKm={search?.radius_km ?? 0} points={points} focus={focus} />
        {center && radius > 0 && (
          <Circle center={center} radius={radius} pathOptions={{ color: '#d9480f', weight: 2, dashArray: '6 4', fillOpacity: 0.12 }} />
        )}
        {center && (
          <CircleMarker center={center} radius={5} pathOptions={{ color: '#fff', weight: 2, fillColor: '#3d1d0a', fillOpacity: 1 }}>
            {search?.label && <Tooltip direction="top" offset={[0, -6]}>around {search.label}</Tooltip>}
          </CircleMarker>
        )}
        {ghostPins.map((p) => (
          <Marker key={`g-${p.place_id}`} position={[p.lat, p.lon]} icon={icon('ghost-pin', '')} zIndexOffset={-100}>
            <Tooltip>{p.name} (dropped)</Tooltip>
          </Marker>
        ))}
        {located.map((p) => {
          const i = places.indexOf(p)
          const isPinned = !!pinned[p.place_id]
          return (
            <Marker key={p.place_id} position={[p.lat, p.lon]}
              icon={icon(`pick-pin${hovered === p.place_id ? ' hot' : ''}`, `${i + 1}${isPinned ? '<span class="pin-badge"></span>' : ''}`)}
              eventHandlers={{ mouseover: () => onHover(p.place_id), mouseout: () => onHover(null) }}>
              <Tooltip>{p.name}</Tooltip>
              {isPinned && <SavedPopup p={p} center={center} onUnpin={onUnpin} />}
            </Marker>
          )
        })}
        {saved.filter((p) => !pickIds.has(p.place_id)).map((p) => (
          <Marker key={`s-${p.place_id}`} position={[p.lat, p.lon]} icon={savedIcon} zIndexOffset={500}>
            <SavedPopup p={p} center={center} onUnpin={onUnpin} />
          </Marker>
        ))}
      </MapContainer>
      {center && <span className="realmap-radius">radius {search.radius_km} km</span>}
      <button type="button" className="realmap-expand" onClick={() => setBig((b) => !b)} aria-label={big ? 'Shrink map' : 'Expand map'}>{big ? '⤡' : '⤢'}</button>
    </div>
  )
}

function SavedPopup({ p, center, onUnpin }) {
  const km = center ? haversineKm(center, [p.lat, p.lon]) : null
  const cuisine = p.cuisine && p.cuisine !== 'nan' ? p.cuisine : null
  return (
    <Popup>
      <div className="pin-pop">
        <b>{p.name}</b>
        <div className="muted">{[cuisine, p.area].filter(Boolean).join(' · ')}</div>
        <div>
          {km != null && <span>{km < 1 ? `${Math.round(km * 1000 / 50) * 50} m` : `${km.toFixed(1)} km`} </span>}
          {money(p) && <span>· {money(p)} </span>}
          {p.rating != null && <span>· {p.rating.toFixed(1)} stars</span>}
        </div>
        <div className="pin-pop-actions">
          <button type="button" className="secondary" onClick={() => onUnpin(p.place_id)}>Remove pin</button>
          <a href={`https://www.google.com/maps/dir/?api=1&destination=${p.lat},${p.lon}`} target="_blank" rel="noreferrer">Directions ↗</a>
        </div>
      </div>
    </Popup>
  )
}
