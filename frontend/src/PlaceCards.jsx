function distance(p) {
  // the ranker's own wording first, so the card matches what the reply says
  const said = p.reasons?.find((r) => r.endsWith(' away'))
  if (said) return said.replace(/ away$/, '')
  const km = p.distance_km
  if (km == null) return null
  return km < 1 ? `${Math.round(km * 1000 / 50) * 50} m` : `${km.toFixed(1)} km`
}

function price(p) {
  const { price_min: lo, price_max: hi } = p
  if (lo == null && hi == null) return null
  const range = lo != null && hi != null && hi !== lo ? `$${lo}–${hi}` : `~$${hi ?? lo}`
  return p.price_source === 'estimate' ? `${range} est.` : range
}

/** One recommendation set, rendered under the bot reply that made it. */
export default function PlaceCards({ places, superseded, hovered, onHover, selectedCard, onSelect, pinned = {}, onTogglePin }) {
  const top = Math.max(...places.map((p) => p.score), 1e-9)
  return (
    <div className={`cards${superseded ? ' superseded' : ''}`} onMouseLeave={() => onHover(null)}>
      {superseded && <div className="superseded-label">earlier picks</div>}
      {places.map((p, i) => {
        const dist = distance(p)
        const cost = price(p)
        const cuisine = p.cuisine && p.cuisine !== 'nan' ? p.cuisine : null
        return (
          <article key={p.place_id} style={{ '--i': i }} tabIndex={0} aria-pressed={selectedCard === p.place_id}
            className={`card${!superseded && hovered === p.place_id ? ' hot' : ''}${selectedCard === p.place_id ? ' selected' : ''}${pinned[p.place_id] ? ' is-pinned' : ''}`}
            onMouseEnter={() => !superseded && onHover(p.place_id)}
            onClick={() => onSelect?.(p)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') onSelect?.(p)
              if (e.key.toLowerCase() === 'p') onTogglePin?.(p)
            }}>
            <span className="rank">{i + 1}</span>
            <div className="card-body">
              <h3>{p.name}</h3>
              <div className="card-sub">{[cuisine, p.area].filter(Boolean).join(' · ')}</div>
              <div className="facts">
                {dist && <span>{dist}</span>}
                {cost && <span>{cost}</span>}
                {p.rating != null && <span>★ {p.rating.toFixed(1)}{p.n_reviews ? <small> ({Math.round(p.n_reviews).toLocaleString()})</small> : null}</span>}
              </div>
              <div className="tags">
                {p.halal && <span className="ptag halal">halal</span>}
                {p.vegetarian && <span className="ptag veg">vegetarian</span>}
                {p.aircon === 1 && <span className="ptag">aircon</span>}
              </div>
              {(selectedCard === p.place_id || pinned[p.place_id]) && (
                <button type="button" className={`pin-btn${pinned[p.place_id] ? ' on' : ''}`}
                  onClick={(e) => { e.stopPropagation(); onTogglePin?.(p) }}>
                  {pinned[p.place_id] ? 'Pinned' : 'Pin to map'}
                </button>
              )}
              <div className="score" title={`ranker score ${p.score}`}><span style={{ width: `${(p.score / top) * 100}%` }} /></div>
            </div>
          </article>
        )
      })}
    </div>
  )
}
