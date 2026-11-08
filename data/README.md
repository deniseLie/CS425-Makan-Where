# data/

| path | what | built by |
|---|---|---|
| `woz_converted_data/` | MultiWOZ 2.2 restaurant dialogues as `[TRACK]` rows | `convert_final.py`, then `relabel_budget.py` |
| `synthetic_data/` | synthetic Singlish `[TRACK]` dialogues, including budget dialogues (amounts said many ways, budgets that change) and dietary dialogues ("no" / "none" / "dun have" answering the dietary question -> `ANY`, `non-vegetarian`, `non-halal`, `no-pork`, changes mid-chat) | `build_synthetic.py` |
| `reply_data/` | synthetic `[REPLY]` rows for the reply generator, replies written by Claude Code (see its README) | `backend/src/generator/drafts.py`, `teacher.py`, `checks.py` |
| `master/places_nea.csv` | NEA establishments, plain conversion | `convert_nea_places.py` |
| `master/places.csv` | recommender place database (cleaned + enriched) | `build_places.py` |
| `master/aliases.csv` | alias table for slot values | hand-edited |
| `raw/` | raw downloads, not committed (gitignored): `NEA_Establishments.xlsx` (source of `places_nea.csv`), `area_region_map.xlsx` (planning area → region), and the optional `osm_food.json`, `muis_halal.csv`, `google_places.csv` read by `build_places.py` | - |

Both `[TRACK]` datasets use the shared tracker format (`User:` / `Normalized:` / `System:` prompt
lines; six slots plus `feedback` in the answer; times as `HH:MM` or a meal word, no day). Check
every row after rebuilding either dataset:

```bash
python data/check_tracker_format.py
```

`budget` is `"<=N"`, the most the user will pay, with the latest amount they said (`"about $15"` ->
`<=15`, `"budget 8-10"` -> `<=10`, `"actually only $8"` -> `<=8`), or `ANY` for no cap. The data used
to keep only four buckets (`below $5`, `below $10`, `$10-20`, `above $20`); `relabel_budget.py`
read the amounts back from the user's lines (bucket top when there was no number, e.g. MultiWOZ's
`cheap` -> `<=10`, `moderate` -> `<=20`, `expensive` -> `ANY`). `build_synthetic.py` now writes
`<=N` itself, so only a rebuilt MultiWOZ conversion needs it again:

```bash
python data/relabel_budget.py --check    # what would change, and rows whose words and label disagree
python data/relabel_budget.py            # rewrite the tracker files in place
```

## places.csv and aliases.csv (owner: Yuhanna)

| file | what | built by |
|---|---|---|
| `master/places_nea.csv` | plain 1:1 conversion of NEA licensed eating establishments (31,029 rows, nothing cleaned) | `convert_nea_places.py` |
| `master/places.csv` | **the place database the recommender reads**: 27,964 cleaned NEA rows + 6,241 OSM-only places = 34,205 | `build_places.py` |
| `master/aliases.csv` | alias table: Singlish / short / misspelt words -> canonical slot values (480 rows) | hand-edited |

```bash
pip install pandas openpyxl scipy
python data/convert_nea_places.py   # data/raw/NEA_Establishments.xlsx -> master/places_nea.csv
python data/build_places.py         # master/places_nea.csv (+ optional data/raw files) -> master/places.csv
```

### What build_places.py does

1. **Cleans** (the rules below, all applied): 6 exact duplicates, 34 repeated licences (keeps the
   company-named row), 2,942 school canteens (university / polytechnic campuses kept), 68
   back-of-house kitchens, 15 staff-only canteens. Pads postal codes to 6 digits, strips stray `?`.
2. **Derives from NEA alone**
   - `setting` from `nea_type` + name keywords: `hawker` (HC, CS, coffeeshops), `food court`
     (Koufu, Food Republic, ..., industrial canteens), `cafe`, `restaurant`, `takeaway`.
   - `aircon`: 1 for mall / MRT / hotel units, 0 for hawker and coffeeshop stalls, blank if unknown.
     The `aircon` setting slot filters out known open-air places and keeps unknown ones.
   - `name`: a cleaned company licensee name ("Sushi Zanmai"), or, when the licensee is a person,
     the stall / unit address ("Stall 01-05 @ Maxwell Food Centre", "Eatery @ Tampines Mall #04-27").
     The reply never shows a stallholder's own name. The original stays in `licensee`.
   - `cuisine` from name keywords and known brands, only when exactly one cuisine matches
     (~1,100 places so far); `tags` keeps finer words (sushi, prata, chicken rice, ...).
   - `vegetarian` when the name says vegetarian / vegan / veggie.
   - `price_min` / `price_max`: a rough band per setting (hawker $3-7 ... restaurant $15-35),
     marked `price_source = estimate`; the reply shows these as "~$6 (est.)".
   - `description`: one generated sentence per place, used by the content ranker.
3. **Enriches from optional files in `data/raw/`** (skipped if missing; matched by location within
   40 m + name / unit number, since NEA names are licensees and stalls have no postal code):

| file | gives | how to get it |
|---|---|---|
| `osm_food.json` | trading name, cuisine, `diet:vegetarian`, opening hours (`diet:halal` only as a tag). OSM places with no NEA match are added as new places (`place_id` `On<id>` / `Ow<id>`; area = most common among the 7 nearest NEA rows; points >1 km from any NEA place dropped) | Overpass query in the docstring of `build_places.py` -> export raw JSON |
| `muis_halal.csv` | halal, and the shop's trading name | CSV export from halal-estab.hzz.me (unofficial directory of MUIS-certified places; columns `Name, Address, Type, Scheme`, postal code at the end of the address). A file with a `postal_code` column also works |
| `google_places.csv` | rating, review count, price range, cuisine, opening hours; unmatched Google places are added as new places (`place_id` `G<id>`) | `python data/fetch_google_places.py` (Google Places API, needs `GOOGLE_MAPS_API_KEY`; setup and cost notes in its docstring; `--dry-run` shows the plan first) |

`halal` is **only** set from the MUIS list, never guessed from a name. OSM's `diet:halal` tag is
unverified, so it only becomes the tag `halal (osm, unverified)`: halal is a hard filter with a
0% violation target.

How the MUIS list is matched (current run: 1,337 entries -> 1,247 halal places; MUIS runs after
OSM, so OSM-only places can be matched too):

- Kitchens, caterers, staff canteens and supermarket halal sections are skipped (522 entries).
- A place is marked halal only when **exactly one** NEA row matches: postal code + unit number,
  then name, then the only NEA row at that postal code. Hawker-centre stalls (no postal code in
  NEA) match on block + street + stall number or on the food-centre name + stall number.
- A MUIS entry that is one stall in a coffeeshop or food court ("#01-357, stall 1") certifies only
  that stall, but NEA lists each stall licence in that unit separately and we can't tell which is
  which. Those rows are **not** marked halal; they get the tag `halal stall` and the stall's name in
  the description (456 entries). If the MUIS name matches one licence holder, that row is halal.
- Where MUIS shows a different shop in the unit than the NEA licence holder (often a newer
  tenant), the MUIS name replaces the name, and cuisine / tags / setting are re-derived from it.
- `halal_evidence` holds the MUIS name and address each halal row was matched to, so anyone can
  spot-check it. Entries that matched nothing are written to `master/muis_unmatched.csv`
  (1,723: mostly places neither NEA nor OSM has at that postal code, or a different unit).
- The source is unofficial; the export date and a spot-check against MUIS's own verify tool
  (halal.muis.gov.sg/verify) should go in the report.

`*_source` columns (`name_source`, `cuisine_source`, `halal_source`, `price_source`) record where
each value came from.

### Columns

`place_id, name, lat, lon, area, cuisine, halal, vegetarian, setting, price_min, price_max, rating,
n_reviews, description, region, address, postal_code, nea_type, licence_number, mrt_exits_500m,
bus_stops_500m, aircon, tags, opening_hours, licensee, building, unit, name_source, cuisine_source,
halal_source, price_source, halal_evidence`

`place_id` is still `P00001`... in NEA spreadsheet order, so ids in gold labels stay valid after a
rebuild. `area` is the NEA planning area (30 of them), `region` one of Central / East / West /
North / North-East; both are lower-cased when the backend loads them.

### aliases.csv

`slot, alias, canonical, lat, lon, note`. Covers every mapping in `backend/src/tracker/values.py`
(a test checks they agree) plus Singlish and short spellings: `kbbq`, `briyani`, `roti canai`,
`ckt`, `amk`, `tpy`, `bugis side`, `tapau`, `kopitiam`, `supper`, `cheap cheap`, `atas`, ...
Misspellings within a small edit distance also match (`chiken rice` -> chinese).

Landmark rows carry `lat, lon`; their planning area is the most common NEA area within 500 m of
that point. Two differ from what the NEA rows suggest and follow `values.py` instead (see `note`):
Clarke Quay (rows around it are mostly `outram`) and Orchard (NEA has almost no Orchard rows).

### Known gaps

- [x] OSM (`data/raw/osm_food.json`, exported 3 Nov 2026): 8,444 named OSM places; 1,896 merged into
      NEA rows (postal code + unit, or name within 200 m), 6,241 added as new places. Cuisine is now
      known for 4,194 places and opening hours for 1,828. Most NEA hawker / coffeeshop stalls are
      licensed to a person and still have no cuisine. An OSM-only place can duplicate an NEA row
      whose licensee name didn't match.
- [x] MUIS halal list -> `data/raw/muis_halal.csv` (1,247 places halal; coverage is partial, see above).
- [ ] Google ratings / prices: run `data/fetch_google_places.py` (default: 12 areas x 10 queries =
      120 requests), then `data/build_places.py`. Until then every place gets the same quality
      score and prices are estimates.
- [ ] Hawker stalls are recommended one by one (5,500 rows). The ranker never shows two stalls from
      the same building in one top 3.
- [ ] NEA has no Orchard, Rochor, Museum, Singapore River or Changi planning areas; places there
      are filed under a neighbouring area.
