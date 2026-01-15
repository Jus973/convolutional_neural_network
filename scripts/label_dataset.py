import os
from enum import IntEnum

class Intent(IntEnum):
    ACKNOWLEDGMENT = 0
    REQUEST        = 1
    INSTRUCTION    = 2
    CLEARANCE      = 3
    REPORT         = 4
    HANDOFF        = 5
    REJECTION      = 6
    OTHER          = 7

INTENT_RULES = [
    (Intent.REQUEST, ["SAY AGAIN", "REPEAT", "CONFIRM", "VERIFY", "READ BACK", "CHECK", "REQUEST", "LOOKING FOR", "CAN WE", "WOULD LIKE", "MAY WE"]),
    (Intent.CLEARANCE, ["CLEARED", "APPROVED", "AUTHORIZED", "CLEARANCE", "LINE UP AND WAIT", "CLEARED TO LAND", "CLEARED FOR TAKEOFF", "ENTER CONTROL ZONE", "ENTER CONTROL AREA"]),
    (Intent.INSTRUCTION, ["CLIMB", "DESCEND", "MAINTAIN", "TURN", "VECTOR", "HOLD", "FOLLOW", "CROSS", "SPEED", "SQUAWK", "EXPEDITE", "EXPECT", "CONTINUE", "PARK", "LAND", "JOIN", "WAIT", "HEADING", "TAKE OFF", "TAXI", "RESUME OWN NAVIGATION", "RESUME OWN NAV", "PROCEED DIRECT", "PROCEED DIRECT TO", "LINE UP RUNWAY", "LINE UP"]),
    (Intent.HANDOFF, ["CONTACT", "MONITOR", "FREQUENCY CHANGE APPROVED", "CHANGE TO", "GO AHEAD ON", "GOOD DAY", "BYE BYE", "CALL PRAHA", "CALL PARIS", "CALL LONDON", "CALL RADAR"]),
    (Intent.REJECTION, ["UNABLE", "CANNOT COMPLY", "NEGATIVE", "DO NOT", "CANNOT", "NO"]),
    (Intent.ACKNOWLEDGMENT, ["ROGER", "WILCO", "AFFIRMATIVE", "AFFIRM", "OK ", "OKAY", "COPY", "COPIED", "THANKS", "THANK YOU", "APPRECIATE"]),
    (Intent.REPORT, ["REPORT", "TRAFFIC", "POSITION", "FLIGHT LEVEL", "ON COURSE", "DIRECT", "LOW ALTITUDE", "AT YOUR", "ELEVEN O'CLOCK", "ONE O'CLOCK", "TWO O'CLOCK", "MILES", "SAME ALTITUDE", "OPPOSITE DIRECTION", "WIND", "QNH", "RUNWAY IS", "RUNWAY THREE TWO IS AVAILABLE", "ESTABLISH ILS", "ESTABLISHED ILS", "RUNWAY IN SIGHT"]),
]

def infer_intent(transcript: str) -> Intent:
    t = transcript.upper()
    for intent, phrases in INTENT_RULES:
        if any(p in t for p in phrases):
            return intent
    return Intent.OTHER


def infer_intent(transcript: str) -> Intent:
    t = transcript.upper()
    
    for intent, phrases in INTENT_RULES:
        if any(p in t for p in phrases):
            return intent
    
    return Intent.OTHER

if __name__ == "__main__":  

    path='./data/spectrograms'
    contents=os.listdir(path)

    num=0
    x=[]

    for fileName in contents:
        stripped=fileName[:-4]
        if (infer_intent(stripped) == Intent.OTHER):
            x.append(stripped)
            num+=1
    
    #add keywords manually looking at OTHER
    x.sort(reverse=True,key=lambda y:len(y))
    print(num)
    for a in x:
        print(a)
        input()

