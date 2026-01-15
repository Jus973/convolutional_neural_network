from enum import Enum

class Intent(str, Enum):
    ACKNOWLEDGMENT = "acknowledgment"
    REQUEST_REPEAT = "request_repeat"
    INSTRUCTION = "instruction"
    CLEARANCE = "clearance"
    REPORT = "report"
    HANDOFF = "handoff"
    REQUEST = "request"
    NEGATIVE = "negative"
    UNABLE = "unable"
    OTHER = "other"


INTENT_RULES = [
    (Intent.REQUEST_REPEAT, ["SAY AGAIN", "REPEAT", "CONFIRM", "VERIFY", "READ BACK", "CHECK"]),
    (Intent.CLEARANCE, ["CLEARED", "APPROVED", "AUTHORIZED", "PROCEED", "LINE UP AND WAIT", "HOLD SHORT"]),
    (Intent.INSTRUCTION, ["CLIMB", "DESCEND", "MAINTAIN", "TURN", "VECTOR", "PROCEED", "HOLD", "FOLLOW", "CROSS", "SPEED", "SQUAWK", "EXPEDITE", "EXPECT"]),
    (Intent.HANDOFF, ["CONTACT", "MONITOR", "FREQUENCY", "STAND BY"]),
    (Intent.REPORT, ["REPORT", "IS AT", "PASSING", "LEAVING", "REACHING", "LEVEL", "MAINTAINING", "OVER", "ESTABLISHED"]),
    (Intent.REQUEST, ["REQUEST", "LOOKING FOR", "CAN WE", "WOULD LIKE"]),
    (Intent.ACKNOWLEDGMENT, ["ROGER", "WILCO", "AFFIRMATIVE", "AFFIRM", "OK", "OKAY", "COPY", "COPIED", "THANKS", "THANK YOU"]),
    (Intent.NEGATIVE, ["NEGATIVE", "DO NOT", "UNABLE", "CANNOT"]),
    (Intent.UNABLE, ["UNABLE", "CANNOT COMPLY"]),
]
def infer_intent(transcript: str) -> Intent:
    t = transcript.upper()
    
    for intent, phrases in INTENT_RULES:
        if any(p in t for p in phrases):
            return intent
    
    return Intent.OTHER

if __name__ == "__main__":  
    examples = [
    "PRAHA LUFTHANSA EIGHT VICTOR LIMA SAY AGAIN",
    "HELLO LUFTHANSA SEVEN EIGHT SEVEN PRAHA RADAR RADAR CONTACT CLIMB FLIGHT LEVEL 120",
    "OKAY COPY THANK YOU",
    "HOTEL GOLF BRAVO IS ECHO TWO AT THREE THOUSAND THREE HUNDRED FEET",
    "CSA SEVEN SEVEN THREE AND HIGHER SPEED APPROVED",
    "ONE TWO SEVEN ONE TWO FIVE CSA SEVEN",
    "ONE TWO FIVE",
    ]

    for e in examples:
        print(e, "→", infer_intent(e))