export const SLOTS = ['cuisine', 'area', 'budget', 'dietary', 'setting', 'time']

function valueText(slot, value, landmark) {
  if (value === 'ANY') return 'any'
  if (slot === 'budget' && value.startsWith('<=')) return `under $${value.slice(2)}`
  if (slot === 'area' && landmark && landmark !== value) return `${landmark} (${value})`
  return value
}

const Row = ({ name, value }) => (
  <div className="slot-row"><span className="k">{name}</span><span className="v">{value}</span></div>
)

/** The tracker's slots as a plain list: what's set so far, then a grey block of what isn't.
 *  (App still passes prevState / critique / turnIdx; they aren't shown.) */
export default function SlotBoard({ state }) {
  const set = SLOTS.filter((s) => state.slots[s] !== 'UNKNOWN')
  const unset = SLOTS.filter((s) => state.slots[s] === 'UNKNOWN')

  return (
    <div className="slot-list">
      {set.map((s) => <Row key={s} name={s} value={valueText(s, state.slots[s], state.landmark)} />)}
      <Row name="radius" value={`${state.radius_km} km`} />
      {unset.length > 0 && (
        <div className="slot-unset" aria-label="Not set yet">
          {unset.map((s) => <Row key={s} name={s} value="not set" />)}
        </div>
      )}
    </div>
  )
}
