"""Deterministic template replies in both styles. Only ever mentions action.places."""

from src.schemas import PolicyAction

SLOT_WORDS = {
    "cuisine": "type of food",
    "area": "area",
    "budget": "budget",
    "dietary": "dietary needs",
    "setting": "kind of place",
    "time": "time",
}


def _opts(options: list[str]) -> str:
    opts = [o.title() for o in options]
    if not opts:
        return ""
    if len(opts) == 1:
        return opts[0]
    return ", ".join(opts[:-1]) + " or " + opts[-1]


def _place_lines(action: PolicyAction) -> str:
    return "\n".join(
        f"{i}. {p.name}" + (f" ({', '.join(p.reasons)})" if p.reasons else "")
        for i, p in enumerate(action.places, 1)
    )


def _ask(action: PolicyAction, singlish: bool) -> str:
    slot, opts = action.ask_slot, _opts(action.ask_options)
    if singlish:
        lines = {
            "cuisine": f"Can! You feel like eating what ah? Got {opts} - or anything also can?",
            "area": f"Ok can! Which side you want to makan? Got {opts}, or anywhere also can?",
            "dietary": f"Got any dietary needs or not? Like {opts}?",
            "budget": "What's your max per person ah?",
            "setting": f"You want hawker or more atas one? Got {opts}.",
            "time": f"You eating when ah? {opts}?",
        }
        fallback = f"Your {SLOT_WORDS.get(slot, slot)} leh? Got {opts}."
    else:
        lines = {
            "cuisine": f"Sure! What kind of food are you in the mood for? For example {opts} - or anything is fine.",
            "area": f"Got it. Which area would you like to eat in? For example {opts}, or anywhere.",
            "dietary": f"Do you have any dietary requirements, such as {opts}?",
            "budget": "What's your maximum budget per person?",
            "setting": f"What kind of place do you prefer - {opts}?",
            "time": f"When are you planning to eat - {opts}?",
        }
        fallback = f"What {SLOT_WORDS.get(slot, slot)} would you like? Options: {opts}."
    reply = lines.get(slot, fallback)
    if not opts:  # no options to offer -> drop the example clause
        reply = reply.split("?")[0] + "?"
    return reply


def template_reply(action: PolicyAction, style: str) -> str:
    singlish = style == "singlish"

    if action.kind == "ask":
        return _ask(action, singlish)

    if not action.places:
        if action.relaxed_slot == "budget":
            return ("Cannot find anything within your budget leh. Want to see the closest options above it?"
                    if singlish else
                    "I couldn't find a match within your budget. Want to see the closest options above it?")
        if action.relaxed_slot == "dietary":
            return ("Cannot confirm which places are non-halal leh. Any halal status okay?"
                    if singlish else
                    "I can't verify which places are non-halal. Would you like to search with any halal status?")
        return ("Aiyo, cannot find anything that fits leh. Want to try another area or cuisine?"
                if singlish else
                "Sorry, I couldn't find anything matching that. Would you like to try another area or cuisine?")

    places = _place_lines(action)
    if action.kind == "fallback_recommend":
        what = SLOT_WORDS.get(action.relaxed_slot, action.relaxed_slot)
        if action.relaxed_slot == "budget":
            head = ("These are the closest above your budget, based on listed price estimates ah:"
                    if singlish else
                    "Here are the closest options above your budget, based on listed price estimates:")
        else:
            head = (f"Cannot find exactly what you want leh, so I don't care about the {what} first. Try these:"
                    if singlish else
                    f"I couldn't find an exact match, so I relaxed the {what}. You could try:")
    else:
        head = "Ok try these lah:" if singlish else "Here are my top picks:"
    tail = "Want something else just tell me ah." if singlish else "Let me know if you'd like something different."
    return f"{head}\n{places}\n{tail}"
