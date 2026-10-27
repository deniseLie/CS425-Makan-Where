# data/

| path | what | built by |
|---|---|---|
| `woz_converted_data/` | MultiWOZ 2.2 restaurant dialogues as `[TRACK]` rows | `convert_final.py` |
| `synthetic_data/` | synthetic Singlish `[TRACK]` dialogues | `build_synthetic.py` |
| `reply_data/` | synthetic `[REPLY]` rows for the reply generator, replies written by Claude Code (see its README) | `backend/src/generator/drafts.py`, `teacher.py`, `checks.py` |
| `master/places.csv` | recommender place database | `convert_nea_places.py` |
| `raw/` | raw downloads, not committed (gitignored): `NEA_Establishments.xlsx` (source of `places.csv`), `area_region_map.xlsx` (planning area → region) | - |

Both `[TRACK]` datasets use the shared tracker format (`User:` / `Normalized:` / `System:` prompt
lines; six slots plus `feedback` in the answer; times as `HH:MM` or a meal word, no day). Check
every row after rebuilding either dataset:

```bash
python data/check_tracker_format.py
```

## places.csv

A plain conversion of NEA licensed eating establishments (`raw/NEA_Establishments.xlsx`): one place
per NEA row (**31,029 rows**), every value copied exactly as it is in the spreadsheet. Nothing is
dropped, deduplicated or cleaned. The deletions and fixes below are suggestions only.

`places.csv` is committed; the spreadsheet is not. To rebuild, put `NEA_Establishments.xlsx` in
`data/raw/` first.

```bash
pip install pandas openpyxl
python data/convert_nea_places.py    # reads data/raw/NEA_Establishments.xlsx, writes data/master/places.csv
```

### ⚠️ Still empty

These recommender columns are **blank in every row**. NEA has no data for them:

| column | where to get it |
|---|---|
| **`cuisine`** | OpenStreetMap `cuisine` tag, Google Maps |
| **`halal`** | MUIS halal-certified establishments list, OSM `diet:halal` |
| **`vegetarian`** | OSM `diet:vegetarian` (only 44 names contain "VEGETARIAN") |
| **`setting`** | derive from `nea_type` (suggested mapping below), refine with OSM `amenity` |
| **`price_min`**, **`price_max`** | Google Maps price level, MakanMetrics |
| **`rating`**, **`n_reviews`** | Google Maps scrape |
| **`description`** | Google Maps, or generated from the other columns |

Partly empty: **`postal_code`** is blank for 5,553 rows, almost all hawker centre and coffeeshop
stalls, whose address reads `<street> BLK <n> Stall No <unit>`.

No column yet for opening hours (needed for the `time` slot); OSM `opening_hours` has them.

Suggested `setting` mapping from `nea_type`: `HC` (hawker centre) and `CS` (coffeeshop) → `hawker`;
`Cafe/Restaurant (Mall)` and `Other F&B` → `restaurant`; `Cafe/Bistro (Shophouse)` → `cafe`
(it mixes cafes, HDB coffeeshops and restaurants); `Industrial Canteen` → `foodcourt`;
`Kiosk/Takeaway` → `takeaway` (not a value the tracker produces).

### Filled from NEA

| column | NEA column | note |
|---|---|---|
| `place_id` | - | `P00001`, `P00002`, ... in spreadsheet row order (licence numbers repeat, so they can't be the id). Re-sorting the xlsx changes the ids. |
| `name` | `licensee_name` | the licence holder, not the shop's name; about 17,000 rows are a person's name |
| `lat`, `lon` | `latitude`, `longitude` | |
| `area` | `PLN_AREA_N` | upper-case URA planning area, e.g. `DOWNTOWN CORE` (30 areas) |
| `region` | `Region` | `Central`, `East`, `West`, `North`, `North-East` |
| `address` | `premises_address` | |
| `postal_code` | `postal_code` | stored as a number in the sheet, so 2,689 codes lost their leading 0 (`38982` is `038982`) |
| `nea_type` | `Type` | `HC` hawker centre stall, `CS` coffeeshop stall, `Cafe/Bistro (Shophouse)`, `Cafe/Restaurant (Mall)`, `Other F&B`, `Industrial Canteen`, `Kiosk/Takeaway` |
| `licence_number` | `licence_number` | |
| `mrt_exits_500m`, `bus_stops_500m` | same | |

### Suggested deletions (not applied)

**3,065 rows**, leaving 27,964. The rules run in this order so each row is counted once; text
rules match the upper-cased `address`.

| delete | rows | how to find them | e.g. |
|---|---|---|---|
| exact duplicates | 6 | same as an earlier row in every column except `place_id` | `P00335` |
| same licence twice | 34 | `licence_number` repeats; the rows differ only in `name` (co-licensees at one address). Keep one per licence, preferably the company name | `P00420` |
| school canteens | 2,942 | address has `SCHOOL` or `JUNIOR COLLEGE`, unless it also has `UNIVERSITY`, `POLYTECHNIC`, `BUSINESS SCHOOL` or `SCHOOL OF MEDICINE` (campus food at NUS, SMU etc. is open to the public) | `P00440` |
| back-of-house kitchens | 68 | address has the word `KITCHEN` or `BANQUET`: hotel main/banquet/production kitchens, MICE kitchens, and 42 units in the Shimei East Kitchen / Gourmet East Kitchen catering buildings. Whole-word match, so `KITCHENER ROAD` is kept | `P00003` |
| staff-only canteens | 15 | address has `STAFF` or `CAMP` | `P00321` |

Review before deleting: 364 caterers (name contains `CATER`). Many only do delivery and events,
but some also have a shopfront.

Fixes that aren't deletions (also not applied):

- Pad `postal_code` to 6 digits (2,689 rows).
- Strip the stray `?` at the end of 156 names (e.g. `CARLTON HOTEL (SINGAPORE) PTE LTD?`).

### Pending

The backend reads this file by default (`backend/src/config.py`). How it lines up with the tracker:

- [x] Blank `halal` / `vegetarian` count as 0: `load_places()` fills them before `.astype(bool)`.
- [x] `area` and `region` are lower-cased on load (`DOWNTOWN CORE` → `downtown core`), which is
      what the tracker's values become after `backend/src/tracker/values.py` translates them
      (landmarks like `Bugis` → `downtown core`). A region slot (`east`) keeps the places in that
      `region`; an area's centre for the radius filter is the average lat/lon of its rows.
- [ ] `setting`: once filled, use the tracker's words (`hawker`, `aircon`, `cafe`, `food court`).
      The `nea_type` mapping above uses other words (`restaurant`, `foodcourt`, `takeaway`).
- [ ] Until `cuisine` is filled, any request that names a cuisine finds no places.

Data quality:

- [ ] `name` is the licensee, not the trading name, so the reply generator would show a
      stallholder's personal name as the place. Trading names have to come from OSM / Google Maps.
- [ ] Matching other sources by postal code + similar name (`docs/plan.md`) won't work well: names
      are licensees and 5,457 of the 5,508 stalls have no postal code. Match on location
      (lat/lon within a few dozen metres, plus unit / stall number) instead.
- [ ] Decide whether to recommend individual hawker stalls (5,508 rows) or the hawker centre as one
      place.
