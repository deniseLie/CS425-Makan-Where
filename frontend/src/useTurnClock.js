import { useEffect, useState } from 'react'

/** Milliseconds since `running` turned true, ticking every 200 ms (0 while idle). */
export default function useTurnClock(running) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!running) { setElapsed(0); return }
    const start = performance.now()
    const id = setInterval(() => setElapsed(performance.now() - start), 200)
    return () => clearInterval(id)
  }, [running])
  return elapsed
}
