import json, sys
p = json.load(open(sys.argv[1]))
for s in p["segments"]:
    if s["text"].startswith("This blanket right here") and abs(s["start"] + s["duration"] - 185.91) < 0.05:
        s["duration"] = round(185.66 - s["start"], 3)
    if s["text"].startswith("And here's the part") and abs(s["start"] - 186.98) < 0.03:
        s["duration"] = round(s["duration"] - (187.05 - s["start"]), 3); s["start"] = 187.05
json.dump(p, open(sys.argv[1], "w"), indent=1)
# the product clone stands at the raw frame's left edge: never push in on him,
# and anchor the medium crop at the left edge so his face stays whole
p = json.load(open(sys.argv[1]))
MED = [1.12, 0.4464, 0.34]
style = {3: MED, 4: MED, 6: None, 8: None, 9: MED, 10: None, 11: MED, 12: None, 13: MED,
         14: None, 15: MED, 17: None, 18: None, 19: None, 20: MED}
for i, f in style.items():
    p["segments"][i]["framing"] = f
json.dump(p, open(sys.argv[1], "w"), indent=1)
