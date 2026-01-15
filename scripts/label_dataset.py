import os
from enum import IntEnum

class Intent(IntEnum):
    ACKNOWLEDGMENT   = 0
    REQUEST_REPEAT   = 1
    INSTRUCTION      = 2
    CLEARANCE        = 3
    REPORT           = 4
    HANDOFF          = 5
    REQUEST          = 6
    NEGATIVE         = 7
    UNABLE           = 8
    OTHER            = 9

INTENT_RULES = [
    (Intent.REQUEST_REPEAT, ["SAY AGAIN", "REPEAT", "CONFIRM", "VERIFY", "READ BACK", "CHECK"]),
    (Intent.CLEARANCE, ["CLEARED", "APPROVED", "AUTHORIZED", "PROCEED", "LINE UP AND WAIT", "HOLD SHORT", "CLEARANCE"]),
    (Intent.INSTRUCTION, ["CLIMB", "DESCEND", "MAINTAIN", "TURN", "VECTOR", "PROCEED", "HOLD", 
                          "FOLLOW", "CROSS", "SPEED", "SQUAWK", "EXPEDITE", "EXPECT", "CONTINUE", "PARK", "LAND", "JOIN"
                          ,"WAIT", "HEADING", "LINE UP", "CLEARED", "HOLD SHORT", "TAKE OFF", "TAXI", "JOIN"]),
    (Intent.HANDOFF, ["CONTACT", "MONITOR", "FREQUENCY", "STAND BY"]),
    (Intent.REQUEST, ["REQUEST", "LOOKING FOR", "CAN WE", "WOULD LIKE"]),
    (Intent.ACKNOWLEDGMENT, ["ROGER", "WILCO", "AFFIRMATIVE", "AFFIRM", "OK", "OKAY", "COPY", "COPIED", "THANKS", "THANK YOU", "APPRECIATE"]),
    (Intent.NEGATIVE, ["NEGATIVE", "DO NOT", "UNABLE", "CANNOT"]),
    (Intent.UNABLE, ["UNABLE", "CANNOT COMPLY"]),
    (Intent.REPORT, ["REPORT", "IS AT", "PASSING", "LEAVING", "REACHING", "LEVEL", "MAINTAINING", "OVER", "ESTABLISHED","AT", "IS"]),
]

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

