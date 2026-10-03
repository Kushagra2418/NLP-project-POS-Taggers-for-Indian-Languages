# POS Tagger for Indian Languages

A rule-based + morphological Part-of-Speech tagger supporting 5 Indian languages.

## Supported Languages
| Code | Language | Script       |
|------|----------|--------------|
| hi   | Hindi    | Devanagari   |
| bn   | Bengali  | Bengali      |
| ta   | Tamil    | Tamil        |
| te   | Telugu   | Telugu       |
| mr   | Marathi  | Devanagari   |

## POS Tags Used (Universal Dependencies)
| Tag   | Description       |
|-------|-------------------|
| NOUN  | Noun              |
| VERB  | Verb              |
| ADJ   | Adjective         |
| ADV   | Adverb            |
| PRON  | Pronoun           |
| AUX   | Auxiliary Verb    |
| ADP   | Postposition      |
| CONJ  | Conjunction       |
| NUM   | Number            |
| PROPN | Proper Noun       |
| PUNCT | Punctuation       |
| X     | Unknown           |

## How It Works

### 3-Layer Tagging Pipeline
1. **Special token detection** — punctuation & numerals (regex)
2. **Lexicon lookup** — hand-crafted lexicons per language (~30–80 entries each)
3. **Morphological suffix matching** — verb/noun/adjective suffixes per language

### Architecture
```
Input Text
    ↓
Tokenizer (indic-nlp-library)
    ↓
For each token:
  ├── Punctuation? → PUNCT
  ├── Number? → NUM
  ├── In lexicon? → tag from lexicon
  ├── Verb suffix? → VERB
  ├── Adjective suffix? → ADJ
  ├── Noun suffix? → NOUN
  └── Default → NOUN
    ↓
Tagged Output [(token, POS), ...]
```

## Installation
```bash
pip install indic-nlp-library nltk
```

## Usage

```python
from pos_tagger import IndianLanguagePOSTagger

# Initialize for Hindi
tagger = IndianLanguagePOSTagger("hi")

# Tag a sentence
result = tagger.tag("मैं आज दिल्ली जाता हूँ")
print(result)
# [('मैं', 'PRON'), ('आज', 'ADV'), ('दिल्ली', 'PROPN'), ('जाता', 'VERB'), ('हूँ', 'AUX')]

# Get detailed output
details = tagger.tag_with_details("मैं आज दिल्ली जाता हूँ")
for item in details:
    print(f"{item['token']:20} {item['pos']:8} {item['description']}")

# Get statistics
stats = tagger.get_stats(details)
print(stats)
```

## Files
- `pos_tagger.py` — Main tagger (lexicons, morphological rules, tagging logic)
- `evaluate.py` — Accuracy evaluation on gold-standard test set
- `README.md` — This file

## Evaluation Results
| Language | Accuracy |
|----------|----------|
| Hindi    | 100%     |
| Bengali  | 100%     |
| Tamil    | 100%     |
| Telugu   | 100%     |
| Marathi  | 87.5%    |
| **Overall** | **97.6%** |

## Extending the Project

### Add more words to the lexicon
```python
HINDI_LEXICON["नया_शब्द"] = "NOUN"  # add to the dict at top of pos_tagger.py
```

### Add a new language
1. Create a new lexicon dict (URDU_LEXICON, etc.)
2. Add verb/noun/adj suffix lists
3. Register in LANGUAGE_CONFIG

### Upgrade to ML-based tagging
For higher accuracy, integrate:
- **Stanza** (Stanford NLP) — supports Hindi & other Indian languages
- **IndicBERT** — transformer model for 12 Indian languages
- **MuRIL** — Google's multilingual model for Indian languages
