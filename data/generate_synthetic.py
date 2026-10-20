"""
Synthetic [TRACK] dialogues covering what MultiWOZ lacks:
dietary / setting / time slots, Singlish phrasing, ANY, revisions, retractions,
area aliases + regions, and feedback on recommendations (too far / too expensive / no).

Same row format as build_dialogues.py, so the files can simply be concatenated.

Usage:
    python build_synthetic.py --n 1500 --out data/
Writes data/dialogues_synth_train.jsonl and data/dialogues_synth_dev.jsonl
(split by dialogue). Do NOT use this for your final test set: hand-write that.

Value conventions (agree with D):
  - cuisine: canonical cuisine OR a dish name ("chicken rice"); D's alias table maps dishes.
  - area   : exactly what the user said: pool area, landmark alias ("Bugis"), or region ("east").
  - budget : below $5 | below $10 | $10-20 | above $20
  - time   : now | breakfast | lunch | dinner | late night   ("supper" -> late night)
  - feedback (about the LAST system recommendation, only for that turn):
             none | too_far | too_expensive | not_interested
    The code (not the tracker) knows which place was rejected, and applies the effect
    (shrink radius / lower budget tier / exclude place).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SLOTS = ["cuisine", "area", "budget", "dietary", "setting", "time"]
UNKNOWN, ANY = "UNKNOWN", "ANY"
REQUIRED = ["cuisine", "area"]

CUISINES = ["Chinese", "Malay", "Indian", "Korean", "Japanese", "Thai", "Western", "Vietnamese"]
DISHES = ["chicken rice", "laksa", "nasi lemak", "bak kut teh", "char kway teow",
          "roti prata", "hokkien mee"]
AREAS = ["Downtown Core", "Bukit Merah", "Bishan", "Toa Payoh", "Geylang", "Kallang",
         "Bukit Timah", "Tanglin", "Outram", "River Valley", "Queenstown", "Novena",
         "Marine Parade", "Tampines", "Bedok", "Pasir Ris", "Clementi", "Bukit Batok",
         "Jurong West", "Jurong East", "Choa Chu Kang", "Bukit Panjang", "Sembawang",
         "Yishun", "Woodlands", "Hougang", "Ang Mo Kio", "Serangoon", "Sengkang", "Punggol"]
# landmarks people actually say. D must add these to the alias table.
ALIASES = ["Bugis", "Orchard", "Chinatown", "Little India", "Holland Village", "Clarke Quay",
           "Katong", "Paya Lebar", "SMU", "NUS", "NTU", "Jurong Point", "Changi", "Vivocity"]
REGIONS = ["central", "east", "west", "north", "north-east"]

# (singlish, standard English). Singlish gets spelling noise later; English stays clean.
CUISINE_T = [
    ("want eat {x}", "I want to eat {x}."),
    ("{x} got?", "Is there any {x}?"),
    ("craving {x} sia", "I am craving {x}."),
    ("go makan {x} can?", "Can we go and eat {x}?"),
    ("{x} lor", "I would like {x}."),
]
AREA_T = [
    ("near {a} side got?", "Is there anything near {a}?"),
    ("{a} area can", "Somewhere around {a} is fine."),
    ("must be at {a} leh", "It must be in {a}."),
    ("{a} side lah", "Around {a}, please."),
]
REGION_T = [
    ("{a} side can", "Somewhere in the {a} is fine."),
    ("anywhere {a} lor", "Anywhere in the {a} is fine."),
]
VALUE_T = {
    "budget": {
        "below $10": [("cheap cheap pls", "Something cheap, please."),
                      ("cheap one can", "A cheap place is fine.")],
        "below $5": [("below five dollar", "Below five dollars."),
                     ("under $5 can?", "Is under $5 possible?")],
        "$10-20": [("ten to twenty dollar ok", "Ten to twenty dollars is fine."),
                   ("budget about $15 lah", "My budget is about $15.")],
        "above $20": [("can atas one", "It can be a fancier place."),
                      ("dun mind pay more", "I do not mind paying more.")],
    },
    "dietary": {
        "halal": [("must be halal hor", "It must be halal."),
                  ("halal only ah", "Halal only, please."),
                  ("need halal one", "It needs to be halal.")],
        "vegetarian": [("got vegetarian anot?", "Is there vegetarian food?"),
                       ("I vegetarian leh", "I am vegetarian.")],
        "vegan": [("must be vegan leh", "It must be vegan.")],
        "no beef": [("cannot eat beef", "I cannot eat beef."),
                    ("no beef pls", "No beef, please.")],
        "no pork": [("no pork one", "No pork, please."),
                    ("cannot pork lah", "I cannot eat pork.")],
    },
    "setting": {
        "hawker": [("hawker centre can", "A hawker centre is fine."),
                   ("want go hawker", "I want to go to a hawker centre.")],
        "aircon": [("got aircon one or not?", "Is it air-conditioned?"),
                   ("must got aircon sia", "It must be air-conditioned.")],
        "cafe": [("cafe vibe can?", "Can it be a cafe?"),
                 ("chill at cafe lor", "Somewhere to relax, like a cafe.")],
        "food court": [("food court also ok", "A food court is fine.")],
    },
    "time": {
        "now": [("open now anot?", "Is it open now?"),
                ("hungry already, can go now?", "I am hungry, can we go now?")],
        "late night": [("supper still open?", "Is it still open for supper?"),
                       ("late night one", "Somewhere open late at night.")],
        "lunch": [("for lunch", "It is for lunch.")],
        "dinner": [("for dinner tonight", "For dinner tonight.")],
        "breakfast": [("breakfast one", "For breakfast.")],
    },
}
ANY_T = {
    "cuisine": [("anything also can", "Anything is fine."), ("dun mind what food", "I do not mind what food.")],
    "area": [("anywhere also can", "Anywhere is fine."), ("location dun matter", "Location does not matter.")],
    "budget": [("price dun matter", "Price does not matter.")],
    "dietary": [("no restriction lah", "I have no dietary restrictions."), ("dun need halal also can", "Halal is not needed.")],
    "setting": [("hawker or aircon also can", "Any setting is fine."), ("dun care where sit", "I do not care where we sit.")],
    "time": [("anytime also can", "Any time is fine.")],
}
REVISE_T = {
    "cuisine": ("actually {new} not {old} lah", "Actually, I want {new}, not {old}."),
    "area": ("change to {new} side lah", "Let's change it to {new} instead."),
}
FEEDBACK_T = {
    "too_far": [("too far lah", "That is too far."),
                ("eh far sia, got nearer one?", "That is far. Is there anything nearer?")],
    "too_expensive": [("ex leh", "That is expensive."),
                      ("too ex lah, got cheaper?", "Too expensive. Is there anything cheaper?")],
    "not_interested": [("dun want this one", "I do not want this one."),
                       ("next one pls", "Next one, please.")],
}
ACCEPT_T = [("ok can", "Okay, that works."), ("shiok, thanks!", "Great, thank you!")]
NAMES = ["Ah Seng Chicken Rice", "Selera Nasi Padang", "Seoul Garden Bistro", "Lau Pa Sat Stall 12",
         "Sunny Cafe", "Kopi Corner", "Warong Pak Ali", "Tian Tian Kitchen", "Little Saigon"]
P_GOAL = {"cuisine": 0.9, "area": 0.85, "budget": 0.4, "dietary": 0.35, "setting": 0.3, "time": 0.3}


def sample_value(slot, rng):
    if rng.random() < 0.08:
        return ANY
    if slot == "cuisine":
        return rng.choice(CUISINES) if rng.random() < 0.6 else rng.choice(DISHES)
    if slot == "area":
        r = rng.random()
        return rng.choice(AREAS) if r < 0.6 else rng.choice(ALIASES) if r < 0.85 else rng.choice(REGIONS)
    return rng.choice(list(VALUE_T[slot]))


def phrase(slot, val, rng):
    if val == ANY:
        return rng.choice(ANY_T[slot])
    if slot == "cuisine":
        sg, en = rng.choice(CUISINE_T)
        return sg.format(x=val.lower()), en.format(x=val)
    if slot == "area":
        sg, en = rng.choice(REGION_T if val in REGIONS else AREA_T)
        return sg.format(a=val), en.format(a=val)
    return rng.choice(VALUE_T[slot][val])


def answer_phrase(slot, val, rng):
    """Reply to a system question: often just the bare value."""
    if val != ANY and slot in REQUIRED and rng.random() < 0.4:
        return (f"{val.lower() if slot == 'cuisine' else val} lor", f"{val}, please.")
    return phrase(slot, val, rng)


def noise(s, rng):
    if rng.random() < 0.5:
        s = s.lower()
    s = s.replace("halal", rng.choice(["halal", "halal", "hallal", "halaal"]))
    if rng.random() < 0.5:
        s = s.replace("please", "pls")
    if rng.random() < 0.5:
        s = s.rstrip(".?!")
    return s


def make_dialogue(idx, rng, english):
    goal = {s: sample_value(s, rng) for s in SLOTS if rng.random() < P_GOAL[s]}
    if not goal:
        goal["cuisine"] = sample_value("cuisine", rng)
    state = {s: UNKNOWN for s in SLOTS}
    hist, rows, last = [], [], [None]

    def system(text):
        hist.append(f"System: {text}")
        last[0] = "system"

    def user(frags, updates, feedback="none"):
        if last[0] == "user":
            system("Noted.")
        state.update(updates)
        en = " ".join(e for _, e in frags)
        sg = en if english else noise(", ".join(s for s, _ in frags), rng)
        if not english and not hist and rng.random() < 0.3:
            sg = rng.choice(["eh ", "hi ", "bro "]) + sg
        hist.append(f"User: {sg}\nNormalized: {en}")
        last[0] = "user"
        rows.append({"id": f"SYN{idx:05d}_{len(hist)}", "task": "[TRACK]", "source": "synthetic",
                     "style": "english" if english else "singlish",
                     "input": "[TRACK]\n" + "\n".join(hist),
                     "output": json.dumps({**state, "feedback": feedback})})

    # 1. opening turn: 1-3 slots
    first = rng.sample(list(goal), k=min(len(goal), rng.choice([1, 2, 2, 3])))
    user([phrase(s, goal[s], rng) for s in first], {s: goal[s] for s in first})

    # 2. system asks for missing required slots
    for s in REQUIRED:
        if state[s] == UNKNOWN:
            if s == "cuisine":
                opts = ", ".join(rng.sample(CUISINES, 3))
                system(rng.choice(["What kind of food would you like?",
                                   f"There is {opts} nearby, which would you like?"]))
            else:
                system("Which area would you like?")
            goal.setdefault(s, ANY)
            user([answer_phrase(s, goal[s], rng)], {s: goal[s]})

    # 3. user volunteers more preferences later
    rest = [s for s in goal if state[s] == UNKNOWN]
    if rest and rng.random() < 0.6:
        more = rng.sample(rest, k=min(len(rest), rng.choice([1, 2])))
        system("Any other preferences?")
        user([phrase(s, goal[s], rng) for s in more], {s: goal[s] for s in more})

    # 4. revision ("actually Chinese not Korean")
    if rng.random() < 0.3:
        cands = [s for s in ("cuisine", "area") if state[s] not in (UNKNOWN, ANY)]
        if cands:
            s = rng.choice(cands)
            old, new = state[s], sample_value(s, rng)
            for _ in range(10):
                if new not in (old, ANY):
                    break
                new = sample_value(s, rng)
            if new not in (old, ANY):
                low = (lambda v: v.lower()) if s == "cuisine" else (lambda v: v)
                sg_t, en_t = REVISE_T[s]
                user([(sg_t.format(new=low(new), old=low(old)), en_t.format(new=new, old=old))],
                     {s: new})

    # 5. retraction of a soft preference ("dun need halal also can")
    if rng.random() < 0.15:
        soft = [s for s in ("dietary", "setting", "time", "budget") if state[s] not in (UNKNOWN, ANY)]
        if soft:
            s = rng.choice(soft)
            user([phrase(s, ANY, rng)], {s: ANY})

    # 6. recommendation + feedback
    def recommend():
        system(f"How about {rng.choice(NAMES)}? It is {rng.choice(['~$4', '~$5', '~$8', '~$12', '~$18'])} "
               f"per person, {rng.choice([150, 300, 450, 800, 1200])} m away.")
    recommend()
    if rng.random() < 0.45:
        fb = rng.choice(list(FEEDBACK_T))
        user([rng.choice(FEEDBACK_T[fb])], {}, fb)
        recommend()
    user([rng.choice(ACCEPT_T)], {}, "none")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="number of dialogues")
    ap.add_argument("--english-share", type=float, default=0.25,
                    help="share of dialogues written in standard English")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": [], "dev": []}
    for i in range(a.n):
        rows = make_dialogue(i, rng, english=rng.random() < a.english_share)
        split_rows["dev" if i % 10 == 0 else "train"] += rows
    for sp, rows in split_rows.items():
        with open(out / f"dialogues_synth_{sp}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # coverage report
    allrows = split_rows["train"] + split_rows["dev"]
    print(f"{a.n} dialogues -> {len(allrows)} rows "
          f"({len(split_rows['train'])} train / {len(split_rows['dev'])} dev)")
    outs = [json.loads(r["output"]) for r in allrows]
    for s in SLOTS:
        c = Counter("value" if o[s] not in (UNKNOWN, ANY) else o[s] for o in outs)
        print(f"  {s:8s} {dict(c)}")
    print("  feedback", dict(Counter(o["feedback"] for o in outs)))
    print("  style   ", dict(Counter(r["style"] for r in allrows)))


if __name__ == "__main__":
    main()