import json
import os

IN_PATH = "data/texts/coco_class_texts.json"
OUT_PATH = "data/texts/coco_class_texts_multi_prompt.json"

def norm(x):
    # handles "person" or ["person"]
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        a = json.load(f)

    cls = [norm(x) for x in a]

    templates = [
        lambda c: c,
        lambda c: f"a photo of {c}",
        lambda c: f"this is {c}",
    ]

    out = [[t(c) for t in templates] for c in cls]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("wrote", OUT_PATH, "n=", len(out), "example=", out[0])

if __name__ == "__main__":
    main()
