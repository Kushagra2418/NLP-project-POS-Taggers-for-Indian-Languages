"""
Evaluation script for the Indian Language POS Tagger
Computes accuracy on a small manually-annotated test set.
"""

from pos_tagger import IndianLanguagePOSTagger

# ─────────────────────────────────────────────
# Manually annotated test sets (gold standard)
# ─────────────────────────────────────────────

TEST_DATA = {
    "hi": [
        ("मैं",    "PRON"),
        ("आज",    "ADV"),
        ("दिल्ली","PROPN"),
        ("जाता",  "VERB"),
        ("हूँ",   "AUX"),
        ("और",    "CONJ"),
        ("वहाँ",  "ADV"),
        ("अच्छा", "ADJ"),
        ("खाना",  "NOUN"),
        ("।",     "PUNCT"),
    ],
    "bn": [
        ("আমি",     "PRON"),
        ("আজ",      "ADV"),
        ("ঢাকা",    "PROPN"),
        ("যাচ্ছি",  "VERB"),
        ("এবং",     "CONJ"),
        ("ভালো",    "ADJ"),
        ("খাবার",   "NOUN"),
        ("।",       "PUNCT"),
    ],
    "ta": [
        ("நான்",         "PRON"),
        ("இன்று",        "ADV"),
        ("சென்னை",       "PROPN"),
        ("செல்கிறேன்",   "VERB"),
        ("மற்றும்",      "CONJ"),
        ("நல்ல",         "ADJ"),
        ("உணவு",         "NOUN"),
        ("சாப்பிடுவேன்", "VERB"),
    ],
    "te": [
        ("నేను",           "PRON"),
        ("ఈరోజు",          "ADV"),
        ("హైదరాబాద్",      "PROPN"),
        ("వెళ్తున్నాను",   "VERB"),
        ("మరియు",          "CONJ"),
        ("మంచి",           "ADJ"),
        ("భోజనం",          "NOUN"),
    ],
    "mr": [
        ("मी",     "PRON"),
        ("आज",    "ADV"),
        ("पुणे",  "PROPN"),
        ("जातो",  "VERB"),
        ("आणि",   "CONJ"),
        ("चांगले","ADJ"),
        ("जेवण",  "NOUN"),
        ("आहे",   "AUX"),
    ],
}


def evaluate(lang_code):
    tagger = IndianLanguagePOSTagger(lang_code)
    test = TEST_DATA[lang_code]
    correct = 0
    results = []
    for token, gold in test:
        predicted = tagger.tag_token(token)
        ok = predicted == gold
        if ok:
            correct += 1
        results.append((token, gold, predicted, "✓" if ok else "✗"))
    accuracy = correct / len(test) * 100
    return accuracy, results


def main():
    print("=" * 65)
    print("  POS Tagger Evaluation")
    print("=" * 65)
    overall_correct = 0
    overall_total = 0

    for lang_code, lang_data in TEST_DATA.items():
        acc, results = evaluate(lang_code)
        overall_correct += int(acc / 100 * len(lang_data))
        overall_total += len(lang_data)

        print(f"\n🌐 {lang_code.upper()} — Accuracy: {acc:.1f}%")
        print(f"  {'Token':<25} {'Gold':<10} {'Predicted':<10} Status")
        print(f"  {'-'*55}")
        for token, gold, pred, status in results:
            print(f"  {token:<25} {gold:<10} {pred:<10} {status}")

    overall = overall_correct / overall_total * 100
    print(f"\n{'='*65}")
    print(f"  Overall Accuracy: {overall:.1f}%  ({overall_correct}/{overall_total})")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
