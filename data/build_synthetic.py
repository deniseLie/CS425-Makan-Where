"""
Synthetic [TRACK] dialogues covering what MultiWOZ lacks:
dietary / setting / time slots, Singlish phrasing, ANY, revisions, retractions,
area aliases + regions, feedback on recommendations (too far / too expensive / no),
and budgets as the amount the user says, including budgets that change mid-chat.

Same row format as convert_final.py, so the files can simply be concatenated.

Usage:
    python data/build_synthetic.py                       # --n 1500 --n-budget 600
Writes data/synthetic_data/dialogues_synth_train.jsonl and dialogues_synth_dev.jsonl
(split by dialogue). Do NOT use this for your final test set: hand-write that.

Three kinds of dialogue, in this order (ids SYN00000.., SYN01500.., SYN02100..):
  - --n general dialogues (make_dialogue). With the default seed the first 1500 are the
    original synthetic set, unchanged apart from their budget labels (see relabel_budget.py).
  - --n-budget budget dialogues (make_budget_dialogue): the budget is always an amount, said
    in many ways ("max $8", "around 15 bucks", "under $12", "$7 per pax", "budget 8-10",
    "fifteen dollar"), and in all but the "varied" ones it changes once: down ("actually
    only $8"; sometimes as "too ex, max $8" after a recommendation), up ("can go up to $25")
    or to no limit ("any budget also can").
  - --n-dietary dietary dialogues (make_dietary_dialogue): "no restriction" answers to the
    bot's dietary question ("no", "none", "nope", "dun have", "anything" -> ANY; a bare "no" to
    "Any other preferences?" changes nothing), non-vegetarian / non-halal / "no-pork", and
    changes mid-chat ("actually non-veg is fine").
The general and budget dialogues come first and are unchanged by later additions.

Value conventions (agree with D):
  - cuisine: canonical cuisine OR a dish name ("chicken rice"); D's alias table maps dishes.
  - area   : exactly what the user said: pool area, landmark alias ("Bugis"), or region ("east").
  - budget : "<=N", the most the user will pay, from the latest amount they said ("about $15"
             -> <=15, "budget 8-10" -> <=10, "actually only $8" -> <=8). No number ("cheap
             cheap pls") -> <=10. No cap ("can atas one", "price dun matter") -> ANY.
  - dietary: halal | vegetarian | vegan | no beef | no pork ("no-pork" too) | non-vegetarian
             | non-halal. No restriction ("no restriction lah", or "no" / "none" / "anything"
             as the answer to the bot's dietary question) -> ANY.
  - time   : now | breakfast | lunch | dinner | late night   ("supper" -> late night)
             or an exact time as 24-hour HH:MM ("7pm can?" -> 19:00). An exact time wins
             over a meal word ("dinner around 7" -> 19:00).
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
# Budget in make_dialogue: a price band is sampled, then a phrase, and the label is what the phrase
# says (singlish, english, label). "cheap" has no number, so it gets the band's top.
BUDGET_T = {
    "below $10": [("cheap cheap pls", "Something cheap, please.", "<=10"),
                  ("cheap one can", "A cheap place is fine.", "<=10")],
    "below $5": [("below five dollar", "Below five dollars.", "<=5"),
                 ("under $5 can?", "Is under $5 possible?", "<=5")],
    "$10-20": [("ten to twenty dollar ok", "Ten to twenty dollars is fine.", "<=20"),
               ("budget about $15 lah", "My budget is about $15.", "<=15")],
    "above $20": [("can atas one", "It can be a fancier place.", ANY),
                  ("dun mind pay more", "I do not mind paying more.", ANY)],
}
# Budget in make_budget_dialogue: an amount N ({n}, or as words {w} / {W}), or a range {lo}-{hi}
# whose top is N. Both sides of a pair must state the same amount.
AMOUNT_T = [
    ("max ${n} can", "A maximum of ${n}, please."),
    ("a maximum of ${n} ah", "A maximum of ${n}."),
    ("budget ${n} only", "My budget is only ${n}."),
    ("under ${n} pls", "Under ${n}, please."),
    ("below ${n} can?", "Can it be below ${n}?"),
    ("around {n} bucks lor", "Around {n} dollars."),
    ("about ${n} per pax lah", "About ${n} per person."),
    ("${n} per pax ok", "${n} per person is okay."),
    ("dun want more than ${n} leh", "I do not want to spend more than ${n}."),
    ("cannot more than ${n} sia", "It cannot be more than ${n}."),
    ("{w} dollar enough", "{W} dollars is enough."),
    ("{w} bucks can already", "{W} dollars is fine."),
    ("my budget only ${n} hor", "My budget is only ${n}."),
    ("${n} max ah", "${n} at most."),
    ("within ${n} can", "Within ${n}, please."),
    ("tight budget, ${n} max", "I am on a tight budget, ${n} at most."),
    ("${n} and below can", "${n} or less is fine."),
    ("up to ${n} only", "Up to ${n} only."),
    ("max {n} dollar per pax", "At most {n} dollars per person."),
    ("budget around ${n}", "My budget is around ${n}."),
    ("${n} can liao", "${n} is fine."),
    ("max {n} can", "A maximum of {n} dollars."),
    ("below {n} pls", "Below {n} dollars, please."),
    ("{n} per pax max", "At most {n} per person."),
    ("budget {n} lor", "My budget is {n} dollars."),
]
RANGE_T = [
    ("budget {lo}-{hi} lah", "My budget is {lo} to {hi} dollars."),
    ("${lo} to ${hi} can", "Between ${lo} and ${hi}, please."),
    ("around ${lo}-{hi} per pax", "About ${lo} to ${hi} per person."),
    ("{wlo} to {whi} dollar ok", "{Wlo} to {whi} dollars is fine."),
    ("${lo}-${hi} ok", "${lo} to ${hi} is okay."),
    ("{lo} to {hi} bucks lor", "{lo} to {hi} dollars."),
]
# A changed budget: lower, higher, or no limit any more.
DOWN_T = [
    ("actually only ${n} can", "Actually, only ${n}."),
    ("eh budget cut to ${n} lah", "Let me lower my budget to ${n}."),
    ("actually max ${n} ah", "Actually, a maximum of ${n}, please."),
    ("wait ah, below ${n} pls", "Wait, below ${n}, please."),
    ("on second thought, under ${n} lor", "On second thought, under ${n}."),
    ("sorry ah, my budget only {n} bucks", "Sorry, my budget is only {n} dollars."),
    ("make it ${n} max lah", "Make it ${n} at most."),
    ("actually can only spend a maximum of ${n} per pax",
     "Actually, I can only spend a maximum of ${n} per person."),
    ("change budget to ${n} can?", "Can we change the budget to ${n}?"),
    ("{w} dollar max can?", "Can it be {w} dollars at most?"),
]
UP_T = [
    ("actually can go up to ${n}", "Actually, I can go up to ${n}."),
    ("ok budget ${n} also can", "A budget of ${n} is fine too."),
    ("can spend up to {n} bucks lah", "I can spend up to {n} dollars."),
    ("nvm, ${n} per pax also ok", "Never mind, ${n} per person is okay."),
    ("budget increase to ${n} lor", "Let me raise my budget to ${n}."),
    ("eh can stretch to ${n}", "I can stretch to ${n}."),
    ("dun mind up to ${n} actually", "Actually, I do not mind up to ${n}."),
    ("my budget can go up to ${n}", "My budget can go up to ${n}."),
]
NO_CAP_T = [
    ("actually any budget also can", "Actually, any budget is fine."),
    ("price dun matter la", "Price does not matter."),
    ("nvm budget, any price also can", "Never mind the budget, any price is fine."),
    ("dun care about price liao", "I do not care about the price anymore."),
    ("no budget limit lah", "I have no budget limit."),
    ("can atas a bit, money no issue", "It can be fancier, money is no issue."),
]
# "too expensive" that also names the new, lower budget (feedback too_expensive, budget <=N)
TOO_EX_T = [
    ("too ex lah, max ${n} can?", "Too expensive. A maximum of ${n}, please."),
    ("ex leh, got below ${n} one?", "That is expensive. Is there anything below ${n}?"),
    ("too ex, my budget only ${n}", "Too expensive, my budget is only ${n}."),
]
AMOUNTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 28, 30, 35, 40, 50]
AMOUNT_W = [2, 3, 4, 5, 5, 6, 5, 6, 3, 5, 3, 3, 5, 2, 2, 3, 4, 2, 3, 1, 2, 1, 1, 1]
BUDGET_KINDS = {"varied": 0.35, "down": 0.30, "up": 0.20, "any": 0.15}
# Dietary dialogues (make_dietary_dialogue). The bot's dietary question, as the reply templates
# (backend/src/generator/templates.py) and other wordings put it:
DIET_Q = ["Do you have any dietary requirements, such as Halal or Vegetarian?",
          "Got any dietary needs or not? Like Halal, Vegetarian?",
          "Any dietary restrictions?", "Do you have any dietary needs?"]
# Answers to that question meaning no restriction (ANY). A bare "no" means this ONLY here:
# after "Any other preferences?" it changes nothing (NOTHING_MORE_T).
NO_RESTRICTION_T = [
    ("no", "No."), ("no lah", "No."), ("no need", "No need."), ("none", "None."),
    ("nope", "Nope."), ("nah", "Nah, none."), ("dun have", "I don't have any."),
    ("don't have leh", "Don't have any."), ("anything", "Anything is fine."),
    ("anything also can", "Anything is fine."), ("can eat everything one", "I can eat everything."),
]
NOTHING_MORE_T = [("no", "No."), ("no lah", "No, that's all."), ("nope", "Nope."),
                  ("nah, that's all", "Nah, that's all."), ("no need", "No need.")]
# Values that are the opposite of the plain word, and the hyphenated "no-pork" (label "no pork",
# which values.py maps to halal like "no pork"): (singlish, english, label). Keys never collide
# with VALUE_T's, so the general dialogues never draw these.
DIET_NEW_T = {
    "non-vegetarian": [("non-veg can", "Non-veg is fine.", "non-vegetarian"),
                       ("i eat meat one, non vegetarian", "I eat meat, non-vegetarian.", "non-vegetarian"),
                       ("non-vegetarian ok", "Non-vegetarian is okay.", "non-vegetarian"),
                       ("non veg lah", "Non-vegetarian, please.", "non-vegetarian")],
    "non-halal": [("non-halal also can", "Non-halal is fine too.", "non-halal"),
                  ("non halal ok one", "Non-halal is okay.", "non-halal"),
                  ("can non-halal", "Non-halal is fine.", "non-halal")],
    "no-pork": [("no-pork pls", "No-pork, please.", "no pork"),
                ("must be no-pork one", "It must be no-pork.", "no pork")],
}
# Mid-chat changes, by the new value.
DIET_CHANGE_T = {
    "non-vegetarian": [("actually non-veg is fine", "Actually, non-veg is fine."),
                       ("eh can eat meat, non-vegetarian ok", "Actually, I can eat meat. Non-vegetarian is okay."),
                       ("change to non vegetarian lah", "Let's change it to non-vegetarian.")],
    "non-halal": [("actually non-halal also can", "Actually, non-halal is fine too."),
                  ("nvm, non halal ok", "Never mind, non-halal is okay.")],
    "vegetarian": [("actually make it vegetarian", "Actually, make it vegetarian."),
                   ("wait, vegetarian pls", "Wait, vegetarian, please.")],
    "halal": [("actually must be halal", "Actually, it must be halal."),
              ("sorry ah, halal one pls", "Sorry, halal, please.")],
    ANY: [("actually no restriction lah", "Actually, I have no dietary restrictions."),
          ("nvm, can eat everything", "Never mind, I can eat everything.")],
}
DIET_CHANGES = [("vegetarian", "non-vegetarian"), ("halal", "non-halal"), ("non-vegetarian", "vegetarian"),
                ("non-halal", "halal"), ("vegetarian", ANY), ("halal", ANY)]
DIETARY_KINDS = {"no restriction": 0.3, "opposite": 0.3, "change": 0.4}
BUDGET_Q = ["How much would you like to spend per person?", "Any budget in mind?",
            "Any other preferences?"]
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty",
         9: "ninety"}
VALUE_T = {
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
                   ("want go hawker", "I want to go to a hawker centre."),
                   ("kopitiam can already", "A coffee shop is fine."),
                   ("coffeeshop also ok lah", "A coffee shop is okay.")],
        "aircon": [("got aircon one or not?", "Is it air-conditioned?"),
                   ("must got aircon sia", "It must be air-conditioned."),
                   ("too hot liao, want indoor one", "It is too hot, I want somewhere indoors.")],
        "cafe": [("cafe vibe can?", "Can it be a cafe?"),
                 ("chill at cafe lor", "Somewhere to relax, like a cafe."),
                 ("wanna go cafe hopping", "I want to go to a cafe.")],
        "food court": [("food court also ok", "A food court is fine."),
                       ("koufu or food republic like that", "A food court like Koufu or Food Republic.")],
        "restaurant": [("proper restaurant can?", "Can it be a proper restaurant?"),
                       ("want sit down restaurant", "I want a sit-down restaurant."),
                       ("restaurant one better", "A restaurant would be better.")],
        "takeaway": [("want tapau only", "I just want takeaway."),
                     ("can dabao anot?", "Can I get it as takeaway?"),
                     ("takeaway one can", "Takeaway is fine.")],
    },
    "time": {
        "now": [("open now anot?", "Is it open now?"),
                ("hungry already, can go now?", "I am hungry, can we go now?")],
        "late night": [("supper still open?", "Is it still open for supper?"),
                       ("late night one", "Somewhere open late at night.")],
        "lunch": [("for lunch", "It is for lunch.")],
        "dinner": [("for dinner tonight", "For dinner tonight.")],
        "breakfast": [("breakfast one", "For breakfast.")],
        # exact times: about a third of time values (see sample_value)
        "19:00": [("7pm can?", "Is 7pm okay?"),
                  ("dinner around 7 lor", "Dinner at around 7 pm.")],
        "12:30": [("12.30 lunch can", "Lunch at 12:30 is fine.")],
        "08:00": [("8am breakfast got?", "Is there breakfast at 8 am?")],
        "23:00": [("after 11pm still open?", "Is it still open after 11 pm?")],
    },
}
EXACT_TIMES = [v for v in VALUE_T["time"] if ":" in v]
MEAL_TIMES = [v for v in VALUE_T["time"] if ":" not in v]
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
PRICES = ["~$4", "~$5", "~$8", "~$12", "~$18"]  # in the system's recommendations
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
    if slot == "time":
        return rng.choice(EXACT_TIMES if rng.random() < 1 / 3 else MEAL_TIMES)
    return rng.choice(list(BUDGET_T if slot == "budget" else VALUE_T[slot]))


def words(n):
    """8 -> "eight", 25 -> "twenty-five" (n < 100)."""
    tens, ones = divmod(n, 10)
    return _ONES[n] if n < 20 else _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def say_amount(templates, n, rng, ranges=()):
    """(singlish, english, "<=n"): n said with one of the templates, or a range up to n."""
    lo = n - rng.choice([1, 2, 3, 5])
    if ranges and lo >= 2 and rng.random() < 0.25:
        sg, en = rng.choice(ranges)
    else:
        sg, en = rng.choice(templates)
    fill = {"n": n, "w": words(n), "W": words(n).capitalize(), "lo": lo, "hi": n,
            "wlo": words(max(lo, 0)), "whi": words(n), "Wlo": words(max(lo, 0)).capitalize()}
    return sg.format(**fill), en.format(**fill), f"<={n}"


def phrase(slot, val, rng):
    """(singlish, english, value): the value the words state, which for a budget band is the
    exact amount the chosen phrase says."""
    if val == ANY:
        return (*rng.choice(ANY_T[slot]), ANY)
    if slot == "cuisine":
        sg, en = rng.choice(CUISINE_T)
        return sg.format(x=val.lower()), en.format(x=val), val
    if slot == "area":
        sg, en = rng.choice(REGION_T if val in REGIONS else AREA_T)
        return sg.format(a=val), en.format(a=val), val
    if slot == "budget":
        if val.startswith("<="):
            return say_amount(AMOUNT_T, int(val[2:]), rng, RANGE_T)
        return rng.choice(BUDGET_T[val])
    if slot == "dietary" and val in DIET_NEW_T:
        return rng.choice(DIET_NEW_T[val])
    return (*rng.choice(VALUE_T[slot][val]), val)


def answer_phrase(slot, val, rng):
    """Reply to a system question: often just the bare value."""
    if val != ANY and slot in REQUIRED and rng.random() < 0.4:
        return (f"{val.lower() if slot == 'cuisine' else val} lor", f"{val}, please.", val)
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


class Chat:
    """A dialogue being written: the prompt so far, the slot state, and one row per user turn."""

    def __init__(self, idx, rng, english):
        self.idx, self.rng, self.english = idx, rng, english
        self.state = {s: UNKNOWN for s in SLOTS}
        self.hist, self.rows, self.last = [], [], None

    def system(self, text):
        self.hist.append(f"System: {text}")
        self.last = "system"

    def user(self, frags, updates, feedback="none"):
        rng, hist = self.rng, self.hist
        if self.last == "user":
            self.system("Noted.")
        self.state.update(updates)
        en = " ".join(e for _, e in frags)
        sg = en if self.english else noise(", ".join(s for s, _ in frags), rng)
        if not self.english and not hist and rng.random() < 0.3:
            sg = rng.choice(["eh ", "hi ", "bro "]) + sg
        hist.append(f"User: {sg}\nNormalized: {en}")
        self.last = "user"
        self.rows.append({"id": f"SYN{self.idx:05d}_{len(hist)}", "task": "[TRACK]", "source": "synthetic",
                          "style": "english" if self.english else "singlish",
                          "input": "[TRACK]\n" + "\n".join(hist),
                          "output": json.dumps({**self.state, "feedback": feedback})})

    def say(self, goal, slots, answer=False):
        """One user message stating these goal slots. Each slot takes the value its phrase states."""
        said = [(answer_phrase if answer else phrase)(s, goal[s], self.rng) for s in slots]
        self.user([(sg, en) for sg, en, _ in said], {s: v for s, (_, _, v) in zip(slots, said)})

    def recommend(self, price=None):
        rng = self.rng
        self.system(f"How about {rng.choice(NAMES)}? It is {price or rng.choice(PRICES)} "
                    f"per person, {rng.choice([150, 300, 450, 800, 1200])} m away.")


def ask_required(chat, goal, also=()):
    """The system asks for each missing required slot. also: slots the first answer adds."""
    rng = chat.rng
    for s in REQUIRED:
        if chat.state[s] == UNKNOWN:
            if s == "cuisine":
                opts = ", ".join(rng.sample(CUISINES, 3))
                chat.system(rng.choice(["What kind of food would you like?",
                                        f"There is {opts} nearby, which would you like?"]))
            else:
                chat.system("Which area would you like?")
            goal.setdefault(s, ANY)
            chat.say(goal, [s, *also], answer=True)
            also = ()


def make_dialogue(idx, rng, english):
    goal = {s: sample_value(s, rng) for s in SLOTS if rng.random() < P_GOAL[s]}
    if not goal:
        goal["cuisine"] = sample_value("cuisine", rng)
    chat = Chat(idx, rng, english)
    state, user = chat.state, chat.user

    # 1. opening turn: 1-3 slots
    first = rng.sample(list(goal), k=min(len(goal), rng.choice([1, 2, 2, 3])))
    chat.say(goal, first)

    # 2. system asks for missing required slots
    ask_required(chat, goal)

    # 3. user volunteers more preferences later
    rest = [s for s in goal if state[s] == UNKNOWN]
    if rest and rng.random() < 0.6:
        more = rng.sample(rest, k=min(len(rest), rng.choice([1, 2])))
        chat.system("Any other preferences?")
        chat.say(goal, more)

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
        # stated, and not as "any" (checked on the goal: "can atas one" is stated, though its label is ANY)
        soft = [s for s in ("dietary", "setting", "time", "budget") if state[s] != UNKNOWN and goal[s] != ANY]
        if soft:
            s = rng.choice(soft)
            chat.say({s: ANY}, [s])

    # 6. recommendation + feedback
    chat.recommend()
    if rng.random() < 0.45:
        fb = rng.choice(list(FEEDBACK_T))
        user([rng.choice(FEEDBACK_T[fb])], {}, fb)
        chat.recommend()
    user([rng.choice(ACCEPT_T)], {}, "none")
    return chat.rows


def amount(rng, lo, hi):
    """A budget amount from lo to hi; everyday amounts ($6-$15) come up most."""
    pool = [(a, w) for a, w in zip(AMOUNTS, AMOUNT_W) if lo <= a <= hi]
    return rng.choices([a for a, _ in pool], weights=[w for _, w in pool])[0]


def make_budget_dialogue(idx, rng, english, kind):
    """A dialogue whose budget is an amount, said in many ways, which (unless kind is "varied")
    changes once: "down" ("actually only $8"), "up" ("can go up to $25") or "any" ("any budget
    also can"). The budget label always follows the latest thing the user said about it."""
    goal = {s: sample_value(s, rng) for s in SLOTS if s != "budget" and rng.random() < P_GOAL[s]}
    lo, hi = {"down": (8, 50), "up": (3, 30)}.get(kind, (3, 50))
    goal["budget"] = f"<={amount(rng, lo, hi)}"
    chat = Chat(idx, rng, english)
    state = chat.state

    # 1. opening: one or two of the other slots, and the budget in about a third of dialogues
    others = [s for s in goal if s != "budget"]
    first = rng.sample(others, k=min(len(others), rng.choice([1, 1, 2])))
    if rng.random() < 0.35 or not first:
        first.append("budget")
    chat.say(goal, first)

    # 2. missing cuisine / area, half the time answered together with the budget
    ask_required(chat, goal, ["budget"] if state["budget"] == UNKNOWN and rng.random() < 0.5 else [])

    # 3. the system asks about the budget if it hasn't come up yet
    if state["budget"] == UNKNOWN:
        chat.system(rng.choice(BUDGET_Q))
        rest = [s for s in goal if state[s] == UNKNOWN and s != "budget"]
        chat.say(goal, ["budget"] + (rng.sample(rest, k=1) if rest and rng.random() < 0.3 else []))

    # 4. the budget changes, and the label follows it
    old = int(state["budget"][2:])
    too_ex = kind == "down" and rng.random() < 0.4
    if too_ex:  # a recommendation over the new budget: "too ex lah, max $8 can?"
        new = amount(rng, 3, old - 2)
        chat.recommend(price=f"~${rng.randint(new + 1, old)}")
        sg, en, value = say_amount(TOO_EX_T, new, rng)
        chat.user([(sg, en)], {"budget": value}, "too_expensive")
    elif kind != "varied":
        chat.system(rng.choice(["Noted. Anything else?", "Okay. Any other preferences?", "Got it."]))
        if kind == "any":
            sg, en = rng.choice(NO_CAP_T)
            value = ANY
        else:
            new = amount(rng, 3, old - 2) if kind == "down" else amount(rng, old + 2, 60)
            sg, en, value = say_amount(DOWN_T if kind == "down" else UP_T, new, rng)
        chat.user([(sg, en)], {"budget": value})

    # 5. maybe one more preference, then recommendation, feedback and accept
    rest = [s for s in goal if state[s] == UNKNOWN]
    if rest and not too_ex and rng.random() < 0.4:
        chat.system("Any other preferences?")
        chat.say(goal, rng.sample(rest, k=1))
    chat.recommend()
    if rng.random() < 0.35:
        fb = rng.choice(list(FEEDBACK_T))
        chat.user([rng.choice(FEEDBACK_T[fb])], {}, fb)
        chat.recommend()
    chat.user([rng.choice(ACCEPT_T)], {}, "none")
    return chat.rows


def make_dietary_dialogue(idx, rng, english, kind):
    """A dialogue about dietary needs. "no restriction": the bot asks and the user says "no",
    "none", "dun have", "anything", ... (ANY). "opposite": non-vegetarian, non-halal, or
    "no-pork" (no pork). "change": a dietary value, then another ("actually non-veg is fine").
    Half the dialogues also answer "Any other preferences?" with a bare "no", which changes nothing."""
    goal = {s: sample_value(s, rng) for s in SLOTS if s != "dietary" and rng.random() < P_GOAL[s]}
    new = None
    if kind == "change":
        goal["dietary"], new = rng.choice(DIET_CHANGES)
    elif kind == "opposite":
        goal["dietary"] = rng.choice(list(DIET_NEW_T))
    chat = Chat(idx, rng, english)
    state = chat.state

    # 1. opening: one or two of the other slots; the dietary value in about a third of dialogues
    others = [s for s in goal if s != "dietary"]
    first = rng.sample(others, k=min(len(others), rng.choice([1, 1, 2])))
    if "dietary" in goal and (rng.random() < 0.35 or not first):
        first.append("dietary")
    if not first:
        goal["cuisine"] = sample_value("cuisine", rng)
        first = ["cuisine"]
    chat.say(goal, first)

    # 2. missing cuisine / area
    ask_required(chat, goal)

    # 3. the bot asks about diet if it hasn't come up
    if state["dietary"] == UNKNOWN:
        chat.system(rng.choice(DIET_Q))
        if kind == "no restriction":
            chat.user([rng.choice(NO_RESTRICTION_T)], {"dietary": ANY})
        else:
            chat.say(goal, ["dietary"], answer=True)

    # 4. the dietary need changes, and the label follows it
    if new:
        chat.system(rng.choice(["Noted. Anything else?", "Okay. Any other preferences?", "Got it."]))
        chat.user([rng.choice(DIET_CHANGE_T[new])], {"dietary": new})

    # 5. "Any other preferences?": another slot, or a bare "no" that changes nothing
    if rng.random() < 0.5:
        chat.system("Any other preferences?")
        rest = [s for s in goal if state[s] == UNKNOWN]
        if rest and rng.random() < 0.4:
            chat.say(goal, rng.sample(rest, k=1))
        else:
            chat.user([rng.choice(NOTHING_MORE_T)], {})

    # 6. recommendation, feedback and accept
    chat.recommend()
    if rng.random() < 0.35:
        fb = rng.choice(list(FEEDBACK_T))
        chat.user([rng.choice(FEEDBACK_T[fb])], {}, fb)
        chat.recommend()
    chat.user([rng.choice(ACCEPT_T)], {}, "none")
    return chat.rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="number of general dialogues")
    ap.add_argument("--n-budget", type=int, default=600,
                    help="number of budget dialogues (amounts said many ways, budget changes)")
    ap.add_argument("--n-dietary", type=int, default=400,
                    help="number of dietary dialogues (no restriction, non-vegetarian / non-halal, changes)")
    ap.add_argument("--english-share", type=float, default=0.25,
                    help="share of dialogues written in standard English")
    ap.add_argument("--out", default=Path(__file__).resolve().parent / "synthetic_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": [], "dev": []}
    kinds, diet_kinds = Counter(), Counter()
    for i in range(a.n + a.n_budget + a.n_dietary):
        english = rng.random() < a.english_share
        if i < a.n:
            rows = make_dialogue(i, rng, english)
        elif i < a.n + a.n_budget:
            kind = rng.choices(list(BUDGET_KINDS), weights=list(BUDGET_KINDS.values()))[0]
            kinds[kind] += 1
            rows = make_budget_dialogue(i, rng, english, kind)
        else:
            kind = rng.choices(list(DIETARY_KINDS), weights=list(DIETARY_KINDS.values()))[0]
            diet_kinds[kind] += 1
            rows = make_dietary_dialogue(i, rng, english, kind)
        split_rows["dev" if i % 10 == 0 else "train"] += rows
    for sp, rows in split_rows.items():
        with open(out / f"dialogues_synth_{sp}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # coverage report
    allrows = split_rows["train"] + split_rows["dev"]
    print(f"{a.n} + {a.n_budget} budget dialogues {dict(kinds)} + {a.n_dietary} dietary dialogues "
          f"{dict(diet_kinds)} -> {len(allrows)} rows "
          f"({len(split_rows['train'])} train / {len(split_rows['dev'])} dev)")
    outs = [json.loads(r["output"]) for r in allrows]
    for s in SLOTS:
        c = Counter("value" if o[s] not in (UNKNOWN, ANY) else o[s] for o in outs)
        print(f"  {s:8s} {dict(c)}")
    caps = Counter(o["budget"] for o in outs if o["budget"].startswith("<="))
    print(f"  budget amounts: {len(caps)} different caps, most common {caps.most_common(6)}")
    print("  dietary values", dict(Counter(o["dietary"] for o in outs if o["dietary"] != UNKNOWN)))
    print("  feedback", dict(Counter(o["feedback"] for o in outs)))
    print("  style   ", dict(Counter(r["style"] for r in allrows)))


if __name__ == "__main__":
    main()