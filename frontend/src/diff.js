/** Word-level diff of two strings (LCS on lowercased words, ignoring punctuation).
 *  -> [{ op: 'same' | 'del' | 'ins', text }] in reading order. */
export function wordDiff(before, after) {
  const a = before.trim().split(/\s+/).filter(Boolean)
  const b = after.trim().split(/\s+/).filter(Boolean)
  const key = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}$]/gu, '')
  const n = a.length, m = b.length
  const L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i][j] = key(a[i]) === key(b[j]) ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  const out = []
  let i = 0, j = 0
  while (i < n && j < m) {
    if (key(a[i]) === key(b[j])) { out.push({ op: 'same', text: b[j] }); i++; j++ }
    else if (L[i + 1][j] >= L[i][j + 1]) out.push({ op: 'del', text: a[i++] })
    else out.push({ op: 'ins', text: b[j++] })
  }
  while (i < n) out.push({ op: 'del', text: a[i++] })
  while (j < m) out.push({ op: 'ins', text: b[j++] })
  return out
}
