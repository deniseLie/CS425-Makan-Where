import json, random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
AREAS = ["near SMU","Bugis","Yishun","Dhoby Ghaut","Orchard","Chinatown",
         "Tampines","Jurong East","Woodlands","Serangoon"]
CUISINES = ["Chinese","Korean","Japanese","Indian","Malay","British",
            "Italian","Thai","Vietnamese","Western"]
DIETARY = ["halal","vegetarian","no beef","no pork"]
SETTING = ["hawker","aircon","café","restaurant"]
BUDGET = ["cheap","moderate","expensive"]
TIME = ["now","lunch","dinner","late night","supper"]

SEEDS = [
        # ---- dietary + area ----
        {
            "id": "diet_halal_area",
            "style": "singlish",
            "user_turns": ["eh i hungry, near smu got any halal food?", "malay can already, must be halal hor"],
            "assistant_hint": "ask for cuisine, then confirm",
            "target": {"cuisine": "Malay", "area": "near SMU", "budget": "UNKNOWN",
                    "dietary": "halal", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "diet_vegetarian_english",
            "style": "english",
            "user_turns": ["I'd like a vegetarian restaurant near Tampines.", "Indian would be good."],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Indian", "area": "Tampines", "budget": "UNKNOWN",
                    "dietary": "vegetarian", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "diet_nobeef_hawker",
            "style": "singlish",
            "user_turns": ["hawker food near chinatown, no beef can?", "chinese"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Chinese", "area": "Chinatown", "budget": "UNKNOWN",
                    "dietary": "no beef", "setting": "hawker", "time": "UNKNOWN"},
        },
        {
            "id": "diet_nopork",
            "style": "singlish",
            "user_turns": ["any place near woodlands no pork one?", "anything also can lah"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "ANY", "area": "Woodlands", "budget": "UNKNOWN",
                    "dietary": "no pork", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },

        # ---- setting (aircon / hawker / cafe) ----
        {
            "id": "setting_aircon",
            "style": "singlish",
            "user_turns": ["want to eat chicken rice but got aircon one or not?", "bugis"],
            "assistant_hint": "ask for area",
            "target": {"cuisine": "Chinese", "area": "Bugis", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "aircon", "time": "UNKNOWN"},
        },
        {
            "id": "setting_hawker",
            "style": "singlish",
            "user_turns": ["hawker can sit one near serangoon?", "malay food"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Malay", "area": "Serangoon", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "hawker", "time": "UNKNOWN"},
        },
        {
            "id": "setting_cafe",
            "style": "singlish",
            "user_turns": ["any cafe near orchard for brunch?", "western can"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Western", "area": "Orchard", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "café", "time": "UNKNOWN"},
        },

        # ---- budget (indirect) ----
        {
            "id": "budget_cheap_cheap",
            "style": "singlish",
            "user_turns": ["cheap cheap food near yishun can?", "anything also can"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "ANY", "area": "Yishun", "budget": "cheap",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "budget_below_five",
            "style": "singlish",
            "user_turns": ["below five dollar meal near bugis?", "chinese"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Chinese", "area": "Bugis", "budget": "cheap",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "budget_expensive_english",
            "style": "english",
            "user_turns": ["I want an expensive Italian place near Orchard.", "Yes, that's fine."],
            "assistant_hint": "confirm",
            "target": {"cuisine": "Italian", "area": "Orchard", "budget": "expensive",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },

        # ---- time (supper / late night / lunch) ----
        {
            "id": "time_supper",
            "style": "singlish",
            "user_turns": ["dur supper still open near dhoby ghaut?", "indian"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Indian", "area": "Dhoby Ghaut", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "late night"},
        },
        {
            "id": "time_lunch",
            "style": "singlish",
            "user_turns": ["lunch time near smu want eat", "japanese"],
            "assistant_hint": "ask for cuisine",
            "target": {"cuisine": "Japanese", "area": "near SMU", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "lunch"},
        },

        # ---- revision (overwrite) ----
        {
            "id": "rev_cuisine",
            "style": "singlish",
            "user_turns": ["i feel like korean food", "actually chinese better, near orchard"],
            "assistant_hint": "ask for area, then accept revision",
            "target": {"cuisine": "Chinese", "area": "Orchard", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "rev_budget",
            "style": "singlish",
            "user_turns": ["italian food near orchard", "ex leh, cheaper one"],
            "assistant_hint": "recommend, then accept critique",
            "target": {"cuisine": "Italian", "area": "Orchard", "budget": "cheap",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
        {
            "id": "rev_area_too_far",
            "style": "singlish",
            "user_turns": ["want japanese near bugis", "too far lah, nearer one"],
            "assistant_hint": "recommend, then accept critique",
            "target": {"cuisine": "Japanese", "area": "Bugis", "budget": "UNKNOWN",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },

        # ---- multi-slot in one turn ----
        {
            "id": "multi_all",
            "style": "singlish",
            "user_turns": ["near jurong east, cheap korean food, halal must be", "and aircon please"],
            "assistant_hint": "confirm all slots",
            "target": {"cuisine": "Korean", "area": "Jurong East", "budget": "cheap",
                    "dietary": "halal", "setting": "aircon", "time": "UNKNOWN"},
        },
        {
            "id": "multi_two",
            "style": "singlish",
            "user_turns": ["thai food near tampines, not too ex can already"],
            "assistant_hint": "confirm",
            "target": {"cuisine": "Thai", "area": "Tampines", "budget": "moderate",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },

        # ---- ANY vs UNKNOWN control ----
        {
            "id": "any_cuisine",
            "style": "singlish",
            "user_turns": ["near bugis any food also can, just hungry", "budget no preference also"],
            "assistant_hint": "confirm ANY",
            "target": {"cuisine": "ANY", "area": "Bugis", "budget": "ANY",
                    "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        },
    ]  

PROMPT = """Generate ONE Singlish restaurant-search dialogue between a USER and an ASSISTANT.
Rules:
- The user speaks Singlish (particles like lah/leh/sia/hor, "can or not", "got ... one", "cheap cheap").
- The user must mention these preferences across the dialogue: {slots}.
- Use ONLY these values: area in {areas}; cuisine in {cuisines}; dietary in {dietary}; setting in {setting}; budget in {budget}; time in {time}.
- 2 to 5 user turns. End when the assistant has enough to recommend.
- Output ONLY a JSON object: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}], "target": {{"cuisine":..,"area":..,"budget":..,"dietary":..,"setting":..,"time":..}}}}
"""

def sample_slots():
    # force dietary OR setting every time, plus 1-2 others
    slots = [random.choice(["dietary","setting"])]
    slots += random.sample(["cuisine","area","budget","time"], k=random.randint(1,2))
    return slots

def main(n_per_seed=30, out="tracker_data/dialogues_singlish_raw.jsonl"):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    with open(out, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            for _ in range(n_per_seed):
                slots = sample_slots()
                prompt = PROMPT.format(slots=slots, areas=AREAS, cuisines=CUISINES,
                                       dietary=DIETARY, setting=SETTING,
                                       budget=BUDGET, time=TIME)
                msgs = [{"role":"system","content":"You generate synthetic training data."},
                        {"role":"user","content":prompt}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                out_ids = model.generate(**tok(text, return_tensors="pt").to(model.device),
                                         max_new_tokens=600, temperature=0.9, do_sample=True)
                raw = tok.decode(out_ids[0][len(tok(text).input_ids):], skip_special_tokens=True)
                f.write(json.dumps({"seed_id": seed["id"], "raw": raw}, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()