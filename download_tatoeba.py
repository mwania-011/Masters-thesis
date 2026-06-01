from datasets import load_dataset
import os

os.makedirs("data/tatoeba", exist_ok=True)

def save_pair(config, filename, reverse=False):
    ds = load_dataset(
        "Helsinki-NLP/tatoeba_mt",
        config,
        split="test",
        trust_remote_code=True,
        verification_mode="no_checks"
    )

    print("Sample keys:", ds[0].keys())

    path = f"data/tatoeba/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        for sample in ds:

            # ✅ UNIVERSAL FORMAT HANDLER
            if "translation" in sample:
                s = sample["translation"]["src"]
                t = sample["translation"]["tgt"]

            elif "source_sentence" in sample:
                s = sample["source_sentence"]
                t = sample["target_sentence"]

            elif "sourceString" in sample:
                s = sample["sourceString"]
                t = sample["targetString"]

            else:
                print("Skipping unknown sample:", sample.keys())
                continue   # don't crash anymore

            s = s.strip()
            t = t.strip()

            if reverse:
                s, t = t, s

            if s and t:
                f.write(s + "\t" + t + "\n")

    print(f"Saved: {path}")

# EN-DE (reverse)
save_pair("deu-eng", "tatoeba.en-de.txt", reverse=True)

# EN-RU (direct)
save_pair("eng-rus", "tatoeba.en-ru.txt")
