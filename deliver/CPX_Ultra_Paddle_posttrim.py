# hand trims the cfg can't express: segment text -> raw end time
import json, sys
D = sys.argv[1]
p = json.load(open(f"{D}/plan.json"))
for s in p["segments"]:
    if s["text"] == "Big deal.":
        s["duration"] = round(265.15 - s["start"], 3)
    if s["text"] == "What's your face made of?":
        end = s["start"] + s["duration"]
        s["start"] = 70.79
        s["duration"] = round(end - 70.79, 3)
    if s["text"] == "Mine came with a cover,":
        s["duration"] = round(141.79 - s["start"], 3)
for i in (4, 7, 13, 14, 15, 16):   # tight shots where the shown paddle hides GEN Z's face -> medium
    p["segments"][i]["framing"] = [1.12, 0.5, 0.34]
json.dump(p, open(f"{D}/plan.json", "w"), indent=1)
print([round(s["duration"], 2) for s in p["segments"][:3]])
