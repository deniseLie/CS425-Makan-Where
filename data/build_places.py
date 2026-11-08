#!/usr/bin/env python3
"""
Build the recommender's place database, data/master/places.csv.

Input:  data/master/places_nea.csv   (plain NEA conversion, from convert_nea_places.py)
Output: data/master/places.csv       (what the backend reads)
"""

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

COLUMNS = [
    "place_id", "name", "lat", "lon", "area", "cuisine", "halal", "vegetarian", "setting",
    "price_min", "price_max", "rating", "n_reviews", "description",
    "region", "address", "postal_code", "nea_type", "licence_number", "mrt_exits_500m", "bus_stops_500m",
    # added by this script
    "aircon", "tags", "opening_hours", "licensee", "building", "unit",
    "name_source", "cuisine_source", "halal_source", "price_source", "halal_evidence",
]

MATCH_M = 40  # metres: how close an OSM / Google point must be to an NEA row to be the same place

# ---------------------------------------------------------------------------------------------
# 1. cleaning (rules from data/README.md, in the same order)
# ---------------------------------------------------------------------------------------------

# campus food open to the public: universities, polytechnics, ITE, School of the Arts (SOTA)
UNI_RX = (r"UNIVERSITY|POLYTECHNIC|BUSINESS SCHOOL|SCHOOL OF MEDICINE|\bSMU\b|SCHOOL OF THE ARTS"
          r"|\bSOTA\b|\bITE\b|INSTITUTE OF TECHNICAL EDUCATION")


def clean(df):
    report = {}
    df = df.copy()
    addr = df["address"].fillna("").str.upper()

    dup = df.drop(columns="place_id").duplicated()
    report["exact duplicates"] = int(dup.sum())
    df, addr = df[~dup], addr[~dup]

    # same licence twice: keep one row per licence, preferring a company name
    is_co = df["name"].fillna("").apply(is_company)
    order = df.assign(_co=is_co).sort_values(["licence_number", "_co"], ascending=[True, False])
    keep = ~order.duplicated("licence_number") | order["licence_number"].isna()
    keep_ids = set(order.loc[keep, "place_id"])
    lic = ~df["place_id"].isin(keep_ids)
    report["same licence twice"] = int(lic.sum())
    df, addr = df[~lic], addr[~lic]

    school = addr.str.contains(r"SCHOOL|JUNIOR COLLEGE") & ~addr.str.contains(UNI_RX)
    report["school canteens"] = int(school.sum())
    df, addr = df[~school], addr[~school]

    kitchen = addr.str.contains(r"\b(?:KITCHEN|BANQUET)\b")
    report["back-of-house kitchens"] = int(kitchen.sum())
    df, addr = df[~kitchen], addr[~kitchen]

    staff = addr.str.contains(r"\b(?:STAFF|CAMP)\b")
    report["staff-only canteens"] = int(staff.sum())
    df = df[~staff]

    # fixes
    pc = pd.to_numeric(df["postal_code"], errors="coerce")
    df["postal_code"] = pc.map(lambda x: f"{int(x):06d}" if pd.notna(x) else "")
    df["name"] = df["name"].fillna("").str.strip().str.rstrip("?").str.strip()
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------------------------
# 2. derived columns
# ---------------------------------------------------------------------------------------------

COMPANY_RX = re.compile(
    r"\b(?:PTE|LTD|LIMITED|LLP|LLC|INC|CORP\w*|COMPANY|CO|ENTERPRISES?|RESTAURANTS?|CAFE|CAFÉ|COFFEE"
    r"|FOODS?|TRADING|HOLDINGS?|GROUP|SERVICES?|INTERNATIONAL|KITCHEN|BAKERY|BAR|BISTRO|CATER\w*"
    r"|MANAGEMENT|INDUSTR\w*|CLUB|HOTELS?|ASSOCIATION|SOCIETY|CONCEPTS?|DINING|EATERY|HOUSE|SHOP"
    r"|STORE|MART|SINGAPORE|ASIA|PACIFIC|DEVELOPMENT|INVESTMENTS?|VENTURES?|BROTHERS|GLOBAL"
    r"|HOSPITALITY|BRANDS?|TEA|NOODLES?|RICE|BBQ|GRILL|BURGERS?|PIZZA|SUSHI|RAMEN|F&B|F & B)\b"
    r"|\(S\)|&|\d"
)
PERSON_RX = re.compile(r"\b(?:BIN|BINTE|BTE|BT|D/O|S/O|A/L|A/P)\b")
LEGAL_RX = re.compile(
    r"\s*(?:\bPTE\.?\s*LTD\.?|\bPRIVATE\s+LIMITED|\bLIMITED|\bLTD\.?|\bLLP|\bPTE\.?|\(S\)|\(SINGAPORE\))\s*",
    re.I,
)


def is_company(name):
    name = str(name).upper()
    if PERSON_RX.search(name):
        return False
    return bool(COMPANY_RX.search(name))


def title(text):
    words = []
    for w in str(text).split():
        if any(ch.isdigit() for ch in w) or w in ("&", "F&B", "BBQ", "KFC", "MRT", "HDB", "SMU", "NUS", "NTU"):
            words.append(w)
        else:  # "(HOLDINGS)" -> "(Holdings)", "MCDONALD'S" -> "Mcdonald's"
            i = next((k for k, ch in enumerate(w) if ch.isalpha()), 0)
            words.append(w[:i] + w[i:].capitalize())
    return " ".join(words)


STALL_RX = re.compile(r"^(?P<loc>.*?)\s+Stall No\s+(?P<unit>\S+)\s*$", re.I)
UNIT_RX = re.compile(r"#\s*(?P<unit>[A-Z0-9][A-Z0-9\-/]*)", re.I)
TAIL_RX = re.compile(r"\s*SINGAPORE\s*\d{5,6}\s*$", re.I)
BLK_RX = re.compile(r"^(?P<street>.*?)\s+BLK\s+(?P<blk>\S+)$", re.I)


def split_address(address):
    """-> (street, unit, building). Handles both NEA address styles:
    '4 TAMPINES CENTRAL 5 #04-27/29 TAMPINES MALL SINGAPORE 529510'
    'TOA PAYOH LOR 8 BLK 210 Stall No 01-10'"""
    address = str(address or "").strip()
    m = STALL_RX.match(address)
    if m:
        loc = m.group("loc").strip()
        b = BLK_RX.match(loc)
        if b:  # "TOA PAYOH LOR 8 BLK 210" -> "Blk 210 Toa Payoh Lor 8"
            loc = f"BLK {b.group('blk')} {b.group('street')}"
        return title(loc), m.group("unit"), title(loc)
    body = TAIL_RX.sub("", address)
    m = UNIT_RX.search(body)
    if not m:
        return title(re.sub(r"\(.*?\)", "", body).strip()), "", ""
    street = re.sub(r"\(.*?\)", "", body[:m.start()]).strip()
    building = re.sub(r"\(.*?\)", "", body[m.end():]).strip(" ,")
    return title(street), m.group("unit"), title(building)


def display_name(row):
    """Trading-style name from NEA alone. Company licensees keep their (cleaned) name; a person's
    name is replaced by where the stall is, so replies never show a stallholder's own name."""
    if is_company(row["licensee"]):
        name = LEGAL_RX.sub(" ", row["licensee"]).strip(" .,-")
        return title(name), "licensee"
    if row["nea_type"] in ("HC", "CS") and row["unit"]:
        return f"Stall {row['unit']} @ {row['building'] or row['street']}", "address"
    where = row["building"] or row["street"]
    unit = f" #{row['unit']}" if row["unit"] else ""
    return f"Eatery @ {where}{unit}", "address"


CAFE_RX = re.compile(r"\b(?:CAFE|CAFÉ|COFFEE|ESPRESSO|ROAST\w*|BAKERY|BAKEHOUSE|PATISSERIE|DESSERTS?"
                     r"|GELATO|BRUNCH|TEA HOUSE|BUBBLE TEA|TOAST BOX|YA KUN|STARBUCKS)\b")
COFFEESHOP_RX = re.compile(r"\b(?:COFFEE ?SHOP|KOPITIAM|EATING HO?U?SE?|EATING HSE|FOOD CENTRE)\b")
FOODCOURT_RX = re.compile(r"\b(?:FOOD ?COURT|KOUFU|FOOD REPUBLIC|FOOD JUNCTION|FOOD OPERA|FOOD VILLAGE"
                          r"|FOODFARE|FOOD FARE|KOPITIAM INVESTMENT)\b")
MALL_RX = re.compile(r"\b(?:MALL|PLAZA|SHOPPING|MRT STATION|JUNCTION|GALLERIA|SQUARE|HOTEL|AIRPORT"
                     r"|TERMINAL|CITY|POINT|CENTREPOINT|VIVOCITY|ION ORCHARD|PARAGON|RAFFLES CITY)\b")


def setting_of(nea_type, licensee, address):
    """-> (setting, aircon). aircon is 1 / 0 / nan (unknown)."""
    name, addr = str(licensee).upper(), str(address).upper()
    if nea_type in ("HC", "CS"):
        return "hawker", 0.0
    if FOODCOURT_RX.search(name):
        return "food court", 1.0 if nea_type == "Cafe/Restaurant (Mall)" or MALL_RX.search(addr) else np.nan
    if nea_type == "Industrial Canteen":
        return "food court", 0.0
    if COFFEESHOP_RX.search(name):
        return "hawker", 0.0
    if nea_type == "Kiosk/Takeaway":
        return "takeaway", np.nan
    in_mall = nea_type == "Cafe/Restaurant (Mall)" or bool(MALL_RX.search(addr))
    aircon = 1.0 if in_mall else np.nan
    if CAFE_RX.search(name):
        return "cafe", aircon
    return "restaurant", aircon


# keyword (regex on the upper-cased name) -> (cuisine, tag)
CUISINE_WORDS = {
    "korean": [r"KOREAN?", r"SEOUL", r"BUSAN", r"K-?BBQ", r"BIBIMBAP", r"KIMCHI", r"HANSIK", r"JJIGAE",
               r"TTEOK\w*", r"BIBIGO", r"4FINGERS", r"NENE CHICKEN", r"SEOUL GARDEN"],
    "japanese": [r"JAPAN\w*", r"SUSHI\w*", r"RAMEN", r"UDON", r"SOBA", r"IZAKAYA", r"YAKITORI", r"TONKATSU",
                 r"KATSU", r"DONBURI", r"TEPPANYAKI", r"YAKINIKU", r"OMAKASE", r"NIPPON", r"TOKYO", r"OSAKA",
                 r"KYOTO", r"HOKKAIDO", r"YOSHINOYA", r"SUKIYA", r"PEPPER LUNCH", r"GYU\w*", r"ICHIBAN",
                 r"IPPUDO", r"AJISEN", r"KEISUKE", r"SAKAE", r"WATAMI", r"MOS BURGER"],
    "thai": [r"THAI", r"BANGKOK", r"SIAM", r"TOM ?YUM", r"MOO ?KATA", r"SANOOK", r"NAKHON", r"PAD THAI"],
    "vietnamese": [r"VIET\w*", r"PHO", r"SAIGON", r"HANOI", r"BANH MI"],
    "indian": [r"INDIAN?", r"PRATA", r"BI?RI?YANI", r"BRIYANI", r"TANDOOR\w*", r"MASALA", r"CHETTINAD",
               r"PUNJAB\w*", r"BOMBAY", r"MUMBAI", r"DELHI", r"MADRAS", r"CHENNAI", r"BANANA LEAF",
               r"MAMAK", r"THOSAI", r"DOSA\w*", r"NAAN", r"KERALA", r"MURTABAK"],
    "malay": [r"MALAY\w*", r"NASI", r"MEE REBUS", r"SATAY", r"PADANG", r"PENYET", r"INDONESIA\w*",
              r"WARUNG", r"WAROENG", r"MINANG", r"SOTO", r"LONTONG", r"ENCIK TAN", r"AYAM"],
    "chinese": [r"CHINESE", r"DIM ?SUM", r"CHICKEN RICE", r"BAK KUT TEH", r"KWAY TEOW", r"HOT ?POT",
                r"STEAMBOAT", r"MALA", r"SZECHUAN", r"SICHUAN", r"CANTONESE", r"TEOCHEW", r"HAINANESE",
                r"HAKKA", r"SHANGHAI", r"BEIJING", r"PEKING", r"XIAO LONG BAO", r"DUMPLINGS?", r"ZI ?CHAR",
                r"TZE ?CHAR", r"YONG TAU FOO", r"WANTON", r"WONTON", r"CHAR SIEW", r"ROAST DUCK", r"DUCK RICE",
                r"PRAWN MEE", r"FISH SOUP", r"BAK CHOR MEE", r"HAIDILAO", r"DIN TAI FUNG", r"CRYSTAL JADE",
                r"PU TIEN", r"TUNG ?LOK", r"IMPERIAL TREASURE", r"JUMBO SEAFOOD", r"SOUP RESTAURANT",
                r"CLAYPOT", r"KWAY CHAP", r"LAKSA"],
    "western": [r"WESTERN", r"BURGERS?", r"PIZZ\w*", r"PASTA\w*", r"STEAK\w*", r"ITALIAN[OA]?", r"FRENCH",
                r"AMERICAN", r"MEXICAN", r"SPANISH", r"TAPAS", r"BRASSERIE", r"TRATTORIA", r"OSTERIA",
                r"MCDONALD'?S", r"KFC", r"BURGER KING", r"SUBWAY", r"SAIZERIYA", r"ASTONS", r"COLLIN'?S",
                r"FISH ?& ?CO", r"FISH AND CHIPS", r"CHICKEN CHOP", r"SWENSEN'?S", r"NANDO'?S", r"POPEYES",
                r"TEXAS CHICKEN", r"PASTAMANIA", r"ARNOLD'?S", r"JOLLIBEE", r"SHAKE SHACK"],
}
CUISINE_RX = {c: re.compile(r"\b(?:" + "|".join(ws) + r")\b") for c, ws in CUISINE_WORDS.items()}
# words that become tags (lowercased); a subset of the above that names a dish or style
TAG_RX = re.compile(r"\b(SUSHI|RAMEN|UDON|SOBA|IZAKAYA|YAKITORI|TONKATSU|KATSU|DONBURI|TEPPANYAKI|YAKINIKU"
                    r"|OMAKASE|K-?BBQ|BBQ|BIBIMBAP|KIMCHI|PRATA|BIRYANI|BRIYANI|TANDOORI|THOSAI|DOSA|MURTABAK"
                    r"|NASI LEMAK|NASI PADANG|NASI|SATAY|MEE REBUS|PENYET|SOTO|LONTONG|LAKSA|DIM ?SUM"
                    r"|CHICKEN RICE|BAK KUT TEH|KWAY TEOW|HOT ?POT|STEAMBOAT|MALA|XIAO LONG BAO|DUMPLINGS?"
                    r"|ZI ?CHAR|TZE ?CHAR|YONG TAU FOO|WANTON|WONTON|CHAR SIEW|ROAST DUCK|DUCK RICE|PRAWN MEE"
                    r"|FISH SOUP|BAK CHOR MEE|CLAYPOT|KWAY CHAP|TOM ?YUM|MOO ?KATA|PHO|BANH MI|BURGERS?|PIZZA"
                    r"|PASTA|STEAK|TAPAS|FISH AND CHIPS|CHICKEN CHOP|SEAFOOD|NOODLES?|BAKERY|DESSERTS?"
                    r"|BUBBLE TEA|GELATO|ICE CREAM|COFFEE|BRUNCH|FRIED CHICKEN|CURRY|VEGETARIAN|VEGAN"
                    r"|ITALIAN|FRENCH|MEXICAN|SPANISH|AMERICAN|INDONESIAN|TAIWANESE|CANTONESE|TEOCHEW"
                    r"|HAINANESE|SICHUAN|SZECHUAN|PERANAKAN|NYONYA)\b")
VEG_RX = re.compile(r"\b(?:VEGETARIAN|VEGAN|VEGGIE|VEG|MEATLESS|PLANT[- ]BASED|SU SHI|SUSHI VEG)\b")


def cuisine_of(name):
    """-> cuisine or '' (only when exactly one cuisine's words appear)."""
    name = str(name).upper()
    hits = [c for c, rx in CUISINE_RX.items() if rx.search(name)]
    return hits[0] if len(hits) == 1 else ""


def tags_of(name):
    found = [t.lower().replace("  ", " ") for t in TAG_RX.findall(str(name).upper())]
    out = []
    for t in found:
        t = {"briyani": "biryani", "k-bbq": "kbbq", "dimsum": "dim sum", "hotpot": "hot pot",
             "tomyum": "tom yum", "mookata": "moo kata", "zichar": "zi char", "tzechar": "zi char",
             "tze char": "zi char", "burgers": "burger", "noodles": "noodle", "dumplings": "dumpling",
             "desserts": "dessert", "szechuan": "sichuan"}.get(t, t)
        if t not in out:
            out.append(t)
    return ";".join(out)


# rough price band (SGD per person) per setting; only used when no real price is known
PRICE_BAND = {
    "hawker": (3, 7), "food court": (5, 10), "takeaway": (4, 10), "cafe": (8, 20), "restaurant": (15, 35),
}

SETTING_WORDS = {"hawker": "hawker stall", "food court": "food court stall", "takeaway": "takeaway kiosk",
                 "cafe": "cafe", "restaurant": "restaurant"}


def describe(row):
    """One-line text profile for content similarity, from whatever columns are filled."""
    parts = []
    kind = SETTING_WORDS.get(row["setting"], "eatery")
    cuisine = f"{row['cuisine'].capitalize()} " if row["cuisine"] else ""
    where = row["building"] or row["street"]
    parts.append(f"{cuisine}{kind} at {where}, {title(row['area'])}.")
    tags = [t for t in str(row["tags"]).split(";") if t and t not in ("halal stall", "halal (osm, unverified)")]
    if tags:
        parts.append("Serves " + ", ".join(tags) + ".")
    stalls = [x for x in str(row.get("halal_stalls", "") or "").split(";") if x]
    if stalls:
        parts.append("Has a halal-certified stall: " + ", ".join(stalls) + ".")
    extra = []
    if row["halal"] == 1:
        extra.append("halal-certified")
    if row["vegetarian"] == 1:
        extra.append("vegetarian")
    if row["aircon"] == 1:
        extra.append("air-conditioned")
    elif row["aircon"] == 0:
        extra.append("open-air")
    if extra:
        parts.append(", ".join(extra).capitalize() + ".")
    return " ".join(parts)


def derive(df):
    df = df.copy()
    df["licensee"] = df["name"]
    parts = df["address"].apply(split_address)
    df["street"] = [p[0] for p in parts]
    df["unit"] = [p[1] for p in parts]
    df["building"] = [p[2] for p in parts]

    names = df.apply(display_name, axis=1)
    df["name"] = [n for n, _ in names]
    df["name_source"] = [s for _, s in names]

    st = [setting_of(t, n, a) for t, n, a in zip(df["nea_type"], df["licensee"], df["address"])]
    df["setting"] = [s for s, _ in st]
    df["aircon"] = [a for _, a in st]

    df["cuisine"] = df["licensee"].apply(cuisine_of)
    df["cuisine_source"] = np.where(df["cuisine"] != "", "name", "")
    df["tags"] = df["licensee"].apply(tags_of)
    df["vegetarian"] = df["licensee"].str.upper().str.contains(VEG_RX).astype(int)
    df["halal"] = 0
    df["halal_source"] = ""
    df["opening_hours"] = ""

    band = df["setting"].map(PRICE_BAND)
    df["price_min"] = [b[0] for b in band]
    df["price_max"] = [b[1] for b in band]
    df["price_source"] = "estimate"
    df["rating"] = np.nan
    df["n_reviews"] = np.nan
    return df


# ---------------------------------------------------------------------------------------------
# 3. enrichment from optional sources
# ---------------------------------------------------------------------------------------------

def to_xy(lat, lon):
    """lat/lon -> metres on a flat plane around Singapore (good to ~0.1% over 50 km)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return np.c_[(lon - 103.8) * 111_320 * np.cos(np.radians(1.35)), (lat - 1.35) * 110_574]


NAME_STOP = re.compile(r"\b(?:PTE|LTD|LIMITED|PRIVATE|THE|RESTAURANT|CAFE|SINGAPORE|S)\b|[^A-Z0-9 ]")


def name_key(name):
    return " ".join(NAME_STOP.sub(" ", str(name).upper()).split())


def name_sim(a, b):
    a, b = name_key(a), name_key(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def unit_key(unit):
    return re.sub(r"[^0-9A-Z]", "", str(unit or "").upper().lstrip("#"))


FAR_NAME_M = 200   # a strong name match may be this far away (NEA points sit at the building centre)


def match_points(df, pts, max_m=MATCH_M):
    """For each row of pts (lat, lon, name, unit[, postcode][, amenity]), the index of the
    matching df row or -1. Evidence, strongest first:
      1. same postal code + unit number, or same postal code + name similarity >= 0.8
      2. within FAR_NAME_M metres and name similarity >= 0.8
      3. within max_m metres and name similarity >= 0.6, or the same unit number
      4. no name / unit evidence: only when exactly one NEA row is within 15 m (never for a food
         court, which is a whole centre rather than one stall)"""
    from scipy.spatial import cKDTree

    tree = cKDTree(to_xy(df["lat"], df["lon"]))
    xy = to_xy(pts["lat"], pts["lon"])
    by_postal = df.groupby("postal_code").groups if "postal_code" in df else {}
    out = []

    def sim(p, j):
        return max(name_sim(p["name"], df.at[j, "licensee"]), name_sim(p["name"], df.at[j, "name"]))

    for i, near in enumerate(tree.query_ball_point(xy, r=FAR_NAME_M)):
        p = pts.iloc[i]
        uk = unit_key(p.get("unit", ""))
        postcode = str(p.get("postcode", "") or "").strip()
        rows = list(by_postal.get(postcode.zfill(6), [])) if postcode.isdigit() else []
        if rows:
            hit = [j for j in rows if uk and unit_key(df.at[j, "unit"]) == uk]
            if len(hit) != 1:
                hit = [j for j in (hit or rows) if sim(p, j) >= 0.8]
            if len(hit) == 1:
                out.append(hit[0])
                continue
        if not near:
            out.append(-1)
            continue
        d = np.hypot(*(to_xy(df["lat"].iloc[near], df["lon"].iloc[near]) - xy[i]).T)
        best, best_score = -1, 0.0
        for j, dist in zip(near, d):
            s = sim(p, j)
            if dist <= max_m:
                if uk and unit_key(df.at[j, "unit"]) == uk:
                    s += 0.6
                ok = s >= 0.6
            else:
                ok = s >= 0.8
            if ok and s - dist / 10_000 > best_score:
                best, best_score = j, s - dist / 10_000
        if best >= 0:
            out.append(best)
            continue
        close = [j for j, dd in zip(near, d) if dd <= 15]
        food_court = str(p.get("amenity", "")) == "food_court"
        out.append(close[0] if len(close) == 1 and not food_court else -1)
    return np.array(out)


OSM_CUISINE = {
    # OSM cuisine tag -> the place list's cuisine. Vague tags (asian, local, singaporean, regional,
    # international, seafood, noodle, barbecue, coffee_shop, ...) stay as tags only.
    "korean": "korean", "korean_bbq": "korean",
    "japanese": "japanese", "sushi": "japanese", "ramen": "japanese", "udon": "japanese",
    "beef_bowl": "japanese", "teppanyaki": "japanese", "yakiniku": "japanese", "izakaya": "japanese",
    "donburi": "japanese", "tonkatsu": "japanese", "japanese_curry": "japanese",
    "thai": "thai", "vietnamese": "vietnamese", "pho": "vietnamese",
    "indian": "indian", "north_indian": "indian", "south_indian": "indian", "biryani": "indian",
    "prata": "indian", "roti_prata": "indian",
    "malay": "malay", "malaysian": "malay", "indonesian": "malay", "nasi_lemak": "malay", "satay": "malay",
    "chinese": "chinese", "cantonese": "chinese", "taiwanese": "chinese", "hongkong": "chinese",
    "hong_kong": "chinese", "dim_sum": "chinese", "hotpot": "chinese", "hot_pot": "chinese",
    "dumpling": "chinese", "dumplings": "chinese", "sichuan": "chinese", "szechuan": "chinese",
    "teochew": "chinese", "hainanese": "chinese", "chicken_rice": "chinese", "bak_kut_teh": "chinese",
    "western": "western", "italian": "western", "pizza": "western", "pasta": "western", "burger": "western",
    "sandwich": "western", "american": "western", "french": "western", "steak_house": "western",
    "mexican": "western", "spanish": "western", "fish_and_chips": "western", "chicken": "western",
    "fried_chicken": "western", "grill": "western", "mediterranean": "western", "brunch": "western",
    "british": "western", "german": "western", "tex-mex": "western", "bagel": "western",
}


def load_osm(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None or not tags.get("name"):
            continue
        rows.append({
            "lat": lat, "lon": lon, "name": tags.get("name:en") or tags["name"],
            "unit": tags.get("addr:unit", ""), "postcode": tags.get("addr:postcode", ""),
            "osm_id": f"{el.get('type', 'n')[0]}{el.get('id', '')}",
            "street": " ".join(x for x in (tags.get("addr:housenumber", ""), tags.get("addr:street", "")) if x),
            "address": " ".join(x for x in (tags.get("addr:housenumber", ""), tags.get("addr:street", ""),
                                            f"#{tags['addr:unit']}" if tags.get("addr:unit") else "",
                                            tags.get("addr:postcode", "")) if x),
            "cuisine_raw": tags.get("cuisine", ""), "amenity": tags.get("amenity", ""),
            "halal": tags.get("diet:halal", ""), "vegetarian": tags.get("diet:vegetarian", ""),
            "opening_hours": tags.get("opening_hours", ""),
        })
    return pd.DataFrame(rows)


def apply_osm(df, osm):
    """Merge OSM into matching NEA rows. Returns the match index per OSM point (-1 = no match)."""
    idx = match_points(df, osm)
    n = 0
    for i, j in enumerate(idx):
        if j < 0:
            continue
        n += 1
        p = osm.iloc[i]
        shop = LEGAL_RX.sub(" ", str(p["name"])).strip(" .,-")
        raw = [c.strip().lower() for c in re.split(r"[;,]", p["cuisine_raw"]) if c.strip()]
        mapped = [OSM_CUISINE[c] for c in raw if c in OSM_CUISINE]
        if p["amenity"] != "food_court" or df.at[j, "setting"] == "food court":
            if name_sim(shop, df.at[j, "licensee"]) < 0.5:
                # a different business from the licence holder: drop guesses made from its name
                df.at[j, "cuisine"] = cuisine_of(shop)
                df.at[j, "cuisine_source"] = "osm name" if df.at[j, "cuisine"] else ""
                df.at[j, "tags"] = tags_of(shop)
            df.at[j, "name"] = shop
            df.at[j, "name_source"] = "osm"
        if VEG_RX.search(shop.upper()):
            df.at[j, "vegetarian"] = 1
        is_court = p["amenity"] == "food_court" or df.at[j, "setting"] == "food court"
        if is_court:  # many stalls, many cuisines: no single cuisine for the place
            df.at[j, "cuisine"] = ""
            df.at[j, "cuisine_source"] = ""
        elif mapped:
            df.at[j, "cuisine"] = mapped[0]
            df.at[j, "cuisine_source"] = "osm"
        if raw:
            old = [t for t in str(df.at[j, "tags"]).split(";") if t]
            df.at[j, "tags"] = ";".join(dict.fromkeys(old + [c.replace("_", " ") for c in raw]))
        if p["halal"] in ("yes", "only"):
            # OSM halal tags are unverified: a tag for the content ranker, not the hard filter
            # (halal = 1 only comes from the MUIS list)
            tags = [t for t in str(df.at[j, "tags"]).split(";") if t]
            df.at[j, "tags"] = ";".join(dict.fromkeys(tags + ["halal (osm, unverified)"]))
        if p["vegetarian"] in ("yes", "only"):
            df.at[j, "vegetarian"] = 1
        if p["opening_hours"]:
            df.at[j, "opening_hours"] = p["opening_hours"]
        if p["amenity"] == "cafe" and df.at[j, "setting"] == "restaurant":
            df.at[j, "setting"] = "cafe"
    return idx


def _cols(frame):
    frame = frame.copy()
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    return frame


# MUIS types that are not a place people eat at
MUIS_SKIP_TYPES = {"catering company", "central kitchen", "staff canteen", "halal section"}
# MUIS types that are usually one stall inside a coffeeshop / food court / food centre
MUIS_STALL_TYPES = {"hawker", "food station", "short term stall(bazaar/road shows)"}
SUBSTALL_RX = re.compile(r"\b(?:STALL|MR\s*\d|S\d{1,2}\b|STALL\s*\d)|\(MR|\(S\d", re.I)
STREET_ABBR = {"AVE": "AVENUE", "ST": "STREET", "RD": "ROAD", "DR": "DRIVE", "CRES": "CRESCENT",
               "LOR": "LORONG", "CTRL": "CENTRAL", "NTH": "NORTH", "STH": "SOUTH", "UPP": "UPPER",
               "JLN": "JALAN", "BT": "BUKIT", "TG": "TANJONG", "CL": "CLOSE", "PL": "PLACE"}
CENTRE_NOISE = re.compile(r"\b(?:FOOD|CENTRE|CENTER|MARKET|HAWKER|COMPLEX|AND|&|THE)\b")


def street_words(text):
    words = re.sub(r"[^A-Z0-9 ]", " ", str(text).upper()).split()
    return {STREET_ABBR.get(w, w) for w in words if not w.isdigit()}


def centre_key(text):
    return " ".join(CENTRE_NOISE.sub(" ", re.sub(r"[^A-Z ]", " ", str(text).upper())).split())


def parse_muis(muis):
    """Normalize a MUIS export. Accepts either a postal_code column or a postal code at the end of
    the address (the halal-estab export: Name, Address, Type, Scheme)."""
    muis = _cols(muis).fillna("")
    addr = muis["address"].astype(str) if "address" in muis else pd.Series("", index=muis.index)
    if "postal_code" in muis:
        pc = muis["postal_code"].astype(str).str.extract(r"(\d{5,6})")[0]
    else:
        pc = addr.str.extract(r"(\d{6})\s*$")[0]
    muis["postal_code"] = pc.fillna("").str.zfill(6).where(pc.notna(), "")
    muis["unit"] = addr.str.extract(r"#\s*([0-9A-Z][0-9A-Z\-/]*)", flags=re.I)[0].fillna("")
    muis["blk"] = addr.str.extract(r"^\s*(\d+[A-Z]?)\b", flags=re.I)[0].fillna("").str.upper()
    after = addr.str.replace(r"^.*?#\s*[0-9A-Z][0-9A-Z\-/]*", "", regex=True, flags=re.I)
    muis["centre"] = after.str.replace(r"\d{6}\s*$", "", regex=True).map(centre_key)
    muis["street"] = addr.str.replace(r"#.*$", "", regex=True)
    mtype = muis["type"].str.lower() if "type" in muis else pd.Series("", index=muis.index)
    muis["skip"] = mtype.isin(MUIS_SKIP_TYPES)
    if "scheme" in muis:
        muis["skip"] |= muis["scheme"].str.lower().str.contains("preparation")
    muis["substall"] = mtype.isin(MUIS_STALL_TYPES) | addr.str.contains(SUBSTALL_RX)
    return muis


def apply_muis(df, muis):
    """Mark MUIS halal-certified places.

    Office / mall / shophouse units: matched on postal code + unit number (then name, then the only
    NEA row at that postal code). NEA hawker-centre stalls have no postal code: matched on block
    number + street words + stall number, or food-centre name + stall number.

    A MUIS entry that is one stall inside a coffeeshop or food court ("#01-357, stall 1") only
    certifies that stall, but NEA lists the whole coffeeshop as one place. Those NEA rows are not
    marked halal; they get the tag "halal stall" and the stall's name in the description.
    A place is only marked halal when exactly one NEA row matches (several licences in one unit
    must be told apart by name). Returns MUIS entry counts:
    {"halal": n, "halal stall": n, "unmatched": n, "skipped": n}.
    """
    muis = parse_muis(muis)
    stats = {"halal": 0, "halal stall": 0, "unmatched": 0, "skipped": int(muis["skip"].sum())}
    by_postal = df.groupby("postal_code").groups
    stalls = df[df["nea_type"].isin(["HC", "CS"])]
    stall_blk = stalls["building"].str.extract(r"^Blk\s+(\S+)", flags=re.I)[0].fillna("").str.upper()
    stall_unit = stalls["unit"].map(unit_key)
    stall_centre = stalls["building"].map(centre_key)
    for col in ("halal_stalls", "halal_evidence"):
        if col not in df:
            df[col] = ""

    def mark(j, m, strong):
        if m["substall"] and df.at[j, "nea_type"] not in ("HC", "CS"):
            df.at[j, "tags"] = ";".join(dict.fromkeys([t for t in str(df.at[j, "tags"]).split(";") if t]
                                                      + ["halal stall"]))
            names = [x for x in str(df.at[j, "halal_stalls"]).split(";") if x]
            df.at[j, "halal_stalls"] = ";".join(dict.fromkeys(names + [title(m["name"])]))
            return "halal stall"
        df.at[j, "halal"] = 1
        df.at[j, "halal_source"] = "muis"
        df.at[j, "halal_evidence"] = f"{m['name']} | {m['address']}"
        if strong and df.at[j, "name_source"] in ("address", "licensee"):
            shop = LEGAL_RX.sub(" ", str(m["name"])).strip(" .,-")
            if name_sim(shop, df.at[j, "licensee"]) < 0.5:
                # a different business from the licence holder (often a newer tenant): cuisine,
                # tags and setting guessed from the licensee name no longer apply
                df.at[j, "cuisine"] = cuisine_of(shop)
                df.at[j, "cuisine_source"] = "muis name" if df.at[j, "cuisine"] else ""
                df.at[j, "tags"] = tags_of(shop)
                st, ac = setting_of(df.at[j, "nea_type"], shop, df.at[j, "address"])
                df.at[j, "setting"], df.at[j, "aircon"] = st, ac
                df.at[j, "price_min"], df.at[j, "price_max"] = PRICE_BAND[st]
            if VEG_RX.search(shop.upper()):
                df.at[j, "vegetarian"] = 1
            df.at[j, "name"] = title(shop)  # MUIS has the trading name
            df.at[j, "name_source"] = "muis"
        return "halal"

    unmatched = []
    for _, m in muis[~muis["skip"]].iterrows():
        hit, strong = [], True
        uk = unit_key(m["unit"])
        rows = list(by_postal.get(m["postal_code"], [])) if m["postal_code"].strip("0") else []
        if rows:
            hit = [j for j in rows if uk and unit_key(df.at[j, "unit"]) == uk]
            if len(hit) > 1:
                # several licences in one unit (a coffeeshop's stalls): the name has to pick one,
                # otherwise none of them is marked halal, only tagged
                named = [j for j in hit if max(name_sim(m["name"], df.at[j, "licensee"]),
                                               name_sim(m["name"], df.at[j, "name"])) >= 0.8]
                if len(named) == 1:
                    hit = named
                    m = {**m, "substall": False}  # the name pins down this stall's own licence
                else:
                    for j in hit:
                        mark(j, {**m, "substall": True}, False)
                    stats["halal stall"] += 1
                    continue
            if not hit:
                hit = [j for j in rows if max(name_sim(m["name"], df.at[j, "licensee"]),
                                              name_sim(m["name"], df.at[j, "name"])) >= 0.8]
            if len(hit) > 1:
                stats["unmatched"] += 1
                continue
            if not hit and len(rows) == 1 and not m["substall"]:
                hit, strong = rows, False
        if not hit and uk:  # hawker-centre stall
            same_unit = stall_unit == uk
            if m["blk"]:
                cand = stalls.index[same_unit & (stall_blk == m["blk"])]
                words = street_words(m["street"])
                hit = [j for j in cand if street_words(df.at[j, "building"]) & words - {"BLK"}]
            if not hit and m["centre"]:
                hit = list(stalls.index[same_unit & (stall_centre == m["centre"])])
            hit = hit if len(hit) == 1 else []
        if not hit:
            stats["unmatched"] += 1
            unmatched.append(m)
            continue
        stats[mark(hit[0], m, strong)] += 1  # hit has exactly one row here
    stats["unmatched_rows"] = pd.DataFrame(unmatched)
    return stats


HAWKER_NAME_RX = re.compile(r"\b(?:FOOD CENTRE|FOOD CENTER|HAWKER|MARKET|KOPITIAM|COFFEE ?SHOP|EATING HOUSE)\b", re.I)


def add_osm_only(df, osm, idx):
    """OSM places that match no NEA row become places of their own (place_id "O<osm id>").
    Planning area / region: the most common among the 7 nearest NEA rows. Points more than 1 km
    from every NEA place are dropped."""
    from scipy.spatial import cKDTree

    new = osm[(idx < 0)].copy()
    new = new[new["name"].astype(str).str.strip() != ""]
    if new.empty:
        return df
    tree = cKDTree(to_xy(df["lat"], df["lon"]))
    k = min(7, len(df))
    dist, nn = tree.query(to_xy(new["lat"], new["lon"]), k=k)
    dist, nn = dist.reshape(len(new), k), nn.reshape(len(new), k)
    # more than 1 km from every NEA place: outside the areas we know (offshore islands, bad points)
    keep = dist[:, 0] <= 1000
    new, nn = new[keep], nn[keep]
    areas = df["area"].to_numpy()
    regions = df["region"].to_numpy()
    rows = []
    for (_, p), neighbours in zip(new.iterrows(), nn):
        area = pd.Series(areas[neighbours]).mode().iloc[0]
        region = pd.Series(regions[neighbours][areas[neighbours] == area]).mode().iloc[0]
        name = LEGAL_RX.sub(" ", str(p["name"])).strip(" .,-")
        raw = [c.strip().lower() for c in re.split(r"[;,]", p["cuisine_raw"]) if c.strip()]
        mapped = [OSM_CUISINE[c] for c in raw if c in OSM_CUISINE]
        if p["amenity"] == "food_court":
            setting = "hawker" if HAWKER_NAME_RX.search(name) or "food centre" in name.lower() else "food court"
        elif p["amenity"] == "cafe" or CAFE_RX.search(name.upper()):
            setting = "cafe"
        elif HAWKER_NAME_RX.search(name):
            setting = "hawker"
        else:
            setting = "restaurant"
        is_court = p["amenity"] == "food_court"
        cuisine = "" if is_court else (mapped[0] if mapped else cuisine_of(name))
        tags = [t for t in tags_of(name).split(";") if t] + [c.replace("_", " ") for c in raw]
        if p["halal"] in ("yes", "only"):
            tags.append("halal (osm, unverified)")
        lo, hi = PRICE_BAND[setting]
        rows.append({
            "place_id": f"O{p['osm_id']}", "name": name, "lat": p["lat"], "lon": p["lon"],
            "area": area, "region": region, "cuisine": cuisine, "halal": 0,
            "vegetarian": int(p["vegetarian"] in ("yes", "only") or bool(VEG_RX.search(name.upper()))),
            "setting": setting, "aircon": 0.0 if setting == "hawker" else np.nan,
            "price_min": lo, "price_max": hi, "rating": np.nan, "n_reviews": np.nan,
            "address": p.get("address", ""), "postal_code": (re.findall(r"\d{6}", str(p.get("postcode", "") or "")) or [""])[0],
            "nea_type": "", "licence_number": np.nan, "street": p.get("street", ""),
            "unit": str(p.get("unit", "") or ""), "building": "", "licensee": name,
            "tags": ";".join(dict.fromkeys(tags)), "opening_hours": p["opening_hours"],
            "name_source": "osm", "cuisine_source": "osm" if cuisine else "", "halal_source": "",
            "price_source": "estimate", "google_description": "", "halal_stalls": "", "halal_evidence": "",
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


GOOGLE_PRICE = {1: (4, 12), 2: (12, 30), 3: (30, 60), 4: (60, 120)}


def read_google(path):
    g = _cols(pd.read_csv(path, dtype={"postcode": str, "unit": str, "google_id": str}))
    g = g.rename(columns={"user_ratings_total": "n_reviews", "title": "name",
                          "latitude": "lat", "longitude": "lon", "lng": "lon"})
    for col in ("unit", "postcode", "cuisine", "category", "opening_hours", "description", "google_id"):
        if col not in g:
            g[col] = ""
    g[["unit", "postcode", "cuisine", "category", "opening_hours", "description", "google_id"]] = \
        g[["unit", "postcode", "cuisine", "category", "opening_hours", "description", "google_id"]].fillna("")
    return g[g["name"].astype(str).str.strip() != ""].reset_index(drop=True)


def _google_price(p):
    if pd.notna(p.get("price_min")) and p.get("price_min") != "":
        lo = float(p["price_min"])
        hi = float(p["price_max"]) if pd.notna(p.get("price_max")) and p.get("price_max") != "" else lo
        return lo, hi
    if pd.notna(p.get("price_level")) and p.get("price_level") != "" and int(float(p["price_level"])) in GOOGLE_PRICE:
        return GOOGLE_PRICE[int(float(p["price_level"]))]
    return None


def _google_cuisine(p):
    c = str(p.get("cuisine", "") or "")
    return c if c in CUISINE_WORDS else cuisine_of(f"{p.get('category', '')} {p['name']}")


GOOGLE_NAME_MIN = 0.5  # name similarity a Google place needs to merge into a place with a real name


def apply_google(df, g):
    """Merge Google rows into matching places. Returns the match index per Google row.
    A Google place only merges into a place with a real name (OSM, MUIS, Google, a company
    licensee) when the names roughly match or the unit number is the same; otherwise a neighbour's cuisine, rating and price
    would be copied onto the wrong shop. Places named after their address have no real name, so
    location alone is enough there. Rejected rows get -1 and become places of their own."""
    idx = match_points(df, g)
    for i, j in enumerate(idx):
        if j < 0:
            continue
        p = g.iloc[i]
        shop = LEGAL_RX.sub(" ", str(p["name"])).strip(" .,-")
        same_unit = bool(unit_key(p.get("unit", ""))) and unit_key(p.get("unit", "")) == unit_key(df.at[j, "unit"])
        similar = max(name_sim(shop, df.at[j, "name"]), name_sim(shop, df.at[j, "licensee"])) >= GOOGLE_NAME_MIN
        if df.at[j, "name_source"] != "address" and not (similar or same_unit):
            idx[i] = -1
            continue
        if df.at[j, "name_source"] in ("address", "licensee"):
            if name_sim(shop, df.at[j, "licensee"]) < 0.5:  # a different business from the licensee
                df.at[j, "cuisine"] = ""
                df.at[j, "tags"] = tags_of(shop)
            df.at[j, "name"] = shop
            df.at[j, "name_source"] = "google"
        for col in ("rating", "n_reviews"):
            if col in p and pd.notna(p[col]) and p[col] != "":
                df.at[j, col] = float(p[col])
        price = _google_price(p)
        if price:
            df.at[j, "price_min"], df.at[j, "price_max"] = price
            df.at[j, "price_source"] = "google"
        c = _google_cuisine(p)
        if c and (pd.isna(df.at[j, "cuisine"]) or not df.at[j, "cuisine"]) and df.at[j, "setting"] != "food court":
            df.at[j, "cuisine"] = c
            df.at[j, "cuisine_source"] = "google"
        if p.get("opening_hours") and not str(df.at[j, "opening_hours"] or ""):
            df.at[j, "opening_hours"] = p["opening_hours"]
        if p.get("description"):
            df.at[j, "google_description"] = str(p["description"])
    return idx


def add_google_only(df, g, idx):
    """Google places that match no existing place become places of their own (place_id "G<id>")."""
    from scipy.spatial import cKDTree

    new = g[idx < 0]
    if new.empty:
        return df
    base = df[df["place_id"].astype(str).str.startswith("P")]  # area / region from NEA rows only
    tree = cKDTree(to_xy(base["lat"], base["lon"]))
    k = min(7, len(base))
    dist, nn = tree.query(to_xy(new["lat"], new["lon"]), k=k)
    dist, nn = dist.reshape(len(new), k), nn.reshape(len(new), k)
    areas, regions = base["area"].to_numpy(), base["region"].to_numpy()
    rows = []
    for (_, p), d, neighbours in zip(new.iterrows(), dist, nn):
        if d[0] > 1000:
            continue
        area = pd.Series(areas[neighbours]).mode().iloc[0]
        region = pd.Series(regions[neighbours][areas[neighbours] == area]).mode().iloc[0]
        name = LEGAL_RX.sub(" ", str(p["name"])).strip(" .,-")
        cat = str(p.get("category", ""))
        if "food court" in cat:
            setting = "hawker" if HAWKER_NAME_RX.search(name) else "food court"
        elif "cafe" in cat or "coffee" in cat or "bakery" in cat:
            setting = "cafe"
        elif HAWKER_NAME_RX.search(name):
            setting = "hawker"
        else:
            setting = "restaurant"
        price = _google_price(p)
        lo, hi = price or PRICE_BAND[setting]
        cuisine = "" if setting == "food court" else _google_cuisine(p)
        pid = re.sub(r"[^A-Za-z0-9]", "", str(p.get("google_id", ""))) or f"{len(rows)}"
        rows.append({
            "place_id": f"G{pid}", "name": name, "lat": p["lat"], "lon": p["lon"], "area": area,
            "region": region, "cuisine": cuisine, "halal": 0,
            "vegetarian": int(bool(VEG_RX.search(name.upper()))), "setting": setting,
            "aircon": 0.0 if setting == "hawker" else np.nan, "price_min": lo, "price_max": hi,
            "rating": float(p["rating"]) if pd.notna(p.get("rating")) and p.get("rating") != "" else np.nan,
            "n_reviews": float(p["n_reviews"]) if pd.notna(p.get("n_reviews")) and p.get("n_reviews") != "" else np.nan,
            "address": str(p.get("address", "") or ""),
            "postal_code": (re.findall(r"\d{6}", str(p.get("postcode", "") or "")) or [""])[0],
            "nea_type": "", "licence_number": np.nan, "street": "", "unit": str(p.get("unit", "") or ""),
            "building": "", "licensee": name, "tags": tags_of(name), "opening_hours": p.get("opening_hours", ""),
            "name_source": "google", "cuisine_source": "google" if cuisine else "", "halal_source": "",
            "price_source": "google" if price else "estimate", "google_description": str(p.get("description", "") or ""),
            "halal_stalls": "", "halal_evidence": "",
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# ---------------------------------------------------------------------------------------------

def build(nea_path, raw_dir, out_path):
    out_path = Path(out_path)
    nea = pd.read_csv(nea_path, dtype={"licence_number": str})
    print(f"NEA rows: {len(nea)}")
    df, report = clean(nea)
    for k, v in report.items():
        print(f"removed {k}: {v}")
    print(f"rows after cleaning: {len(df)}")

    df = derive(df)
    df["google_description"] = ""
    df["halal_stalls"] = ""
    df["halal_evidence"] = ""

    raw_dir = Path(raw_dir)
    osm_path, muis_path, g_path = raw_dir / "osm_food.json", raw_dir / "muis_halal.csv", raw_dir / "google_places.csv"
    if osm_path.exists():
        osm = load_osm(osm_path)
        idx = apply_osm(df, osm)
        print(f"OSM places: {len(osm)}, merged into NEA rows: {int((idx >= 0).sum())}")
        n_before = len(df)
        df = add_osm_only(df, osm, idx)
        print(f"OSM places added as new places: {len(df) - n_before}")
    else:
        print(f"skipped OSM (no {osm_path})")
    if g_path.exists():
        g = read_google(g_path)
        idx = apply_google(df, g)
        print(f"Google places: {len(g)}, merged into existing places: {int((idx >= 0).sum())}")
        n_before = len(df)
        df = add_google_only(df, g, idx)
        print(f"Google places added as new places: {len(df) - n_before}")
    else:
        print(f"skipped Google (no {g_path})")
    if muis_path.exists():
        stats = apply_muis(df, pd.read_csv(muis_path, dtype=str))
        print(f"MUIS: {stats['halal']} places marked halal, {stats['halal stall']} coffeeshops / food courts "
              f"tagged 'halal stall', {stats['unmatched']} entries not matched, {stats['skipped']} skipped "
              f"(kitchens, caterers, staff canteens)")
        if len(stats["unmatched_rows"]):
            review = out_path.parent / "muis_unmatched.csv"
            stats["unmatched_rows"][["name", "address", "postal_code", "unit"]].to_csv(review, index=False)
            print(f"unmatched MUIS entries listed in {review}")
    else:
        print(f"skipped MUIS (no {muis_path})")

    df["description"] = df.apply(describe, axis=1)
    has_g = df["google_description"].astype(str).str.len() > 0
    df.loc[has_g, "description"] = df.loc[has_g, "description"] + " " + df.loc[has_g, "google_description"]

    df = df.reindex(columns=COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"places written: {len(df)}")
    print("setting:", df["setting"].value_counts().to_dict())
    print(f"with cuisine: {(df['cuisine'] != '').sum()}  "
          + str(df.loc[df['cuisine'] != '', 'cuisine'].value_counts().to_dict()))
    print(f"halal: {int(df['halal'].sum())}  vegetarian: {int(df['vegetarian'].sum())}")
    print(f"aircon known: {int(df['aircon'].notna().sum())}  with rating: {int(df['rating'].notna().sum())}"
          f"  with opening hours: {int((df['opening_hours'].fillna('') != '').sum())}")
    print("name source:", df["name_source"].value_counts().to_dict())
    print(f"Output: {out_path.resolve()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean and enrich places_nea.csv into places.csv.")
    parser.add_argument("--nea", type=Path, default=HERE / "master" / "places_nea.csv")
    parser.add_argument("--raw", type=Path, default=HERE / "raw")
    parser.add_argument("--out", type=Path, default=HERE / "master" / "places.csv")
    args = parser.parse_args()
    build(args.nea, args.raw, args.out)


if __name__ == "__main__":
    main()