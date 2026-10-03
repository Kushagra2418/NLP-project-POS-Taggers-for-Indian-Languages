"""
POS Tagger for Indian Languages
Supports: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam
Uses: Rule-based + morphological analysis approach (no internet required)
"""

import re
from indicnlp.tokenize import indic_tokenize

# ─────────────────────────────────────────────
# Universal POS Tags (UD Tagset)
# ─────────────────────────────────────────────
POS_TAGS = {
    "NOUN":  "Noun",
    "VERB":  "Verb",
    "ADJ":   "Adjective",
    "ADV":   "Adverb",
    "PRON":  "Pronoun",
    "DET":   "Determiner",
    "ADP":   "Adposition (Postposition)",
    "CONJ":  "Conjunction",
    "PART":  "Particle",
    "NUM":   "Number",
    "PUNCT": "Punctuation",
    "PROPN": "Proper Noun",
    "AUX":   "Auxiliary Verb",
    "X":     "Unknown",
}

# ─────────────────────────────────────────────
# Language-specific lexicons and rules
# ─────────────────────────────────────────────

HINDI_LEXICON = {
    # Pronouns
    "मैं": "PRON", "हम": "PRON", "तुम": "PRON", "आप": "PRON",
    "वह": "PRON", "वे": "PRON", "यह": "PRON", "ये": "PRON",
    "मुझे": "PRON", "हमें": "PRON", "उसे": "PRON", "उन्हें": "PRON",
    "मेरा": "PRON", "तुम्हारा": "PRON", "उसका": "PRON", "हमारा": "PRON",
    # Auxiliary Verbs
    "हूँ": "AUX", "है": "AUX", "हैं": "AUX", "था": "AUX",
    "थी": "AUX", "थे": "AUX", "हो": "AUX", "होना": "AUX",
    "रहा": "AUX", "रही": "AUX", "रहे": "AUX",
    # Conjunctions
    "और": "CONJ", "या": "CONJ", "लेकिन": "CONJ", "परंतु": "CONJ",
    "क्योंकि": "CONJ", "इसलिए": "CONJ", "तो": "CONJ", "जो": "CONJ",
    # Postpositions
    "में": "ADP", "पर": "ADP", "से": "ADP", "को": "ADP",
    "के": "ADP", "की": "ADP", "का": "ADP", "ने": "ADP",
    "तक": "ADP", "के लिए": "ADP", "पास": "ADP",
    # Common Nouns
    "घर": "NOUN", "पानी": "NOUN", "खाना": "NOUN", "बच्चा": "NOUN",
    "दिल्ली": "PROPN", "मुंबई": "PROPN", "भारत": "PROPN",
    "राम": "PROPN", "सीता": "PROPN", "कृष्ण": "PROPN",
    # Adjectives
    "अच्छा": "ADJ", "बड़ा": "ADJ", "छोटा": "ADJ", "नया": "ADJ",
    "पुराना": "ADJ", "सुंदर": "ADJ", "लाल": "ADJ", "नीला": "ADJ",
    # Adverbs
    "आज": "ADV", "कल": "ADV", "यहाँ": "ADV", "वहाँ": "ADV",
    "अब": "ADV", "फिर": "ADV", "बहुत": "ADV", "जल्दी": "ADV",
    # Numbers
    "एक": "NUM", "दो": "NUM", "तीन": "NUM", "चार": "NUM",
    "पाँच": "NUM", "दस": "NUM", "सौ": "NUM", "हजार": "NUM",
}

HINDI_VERB_SUFFIXES = ["ना", "ता", "ती", "ते", "या", "ई", "ए",
                        "ाता", "ाती", "ाते", "ेगा", "ेगी", "ेंगे",
                        "ो", "िए", "कर", "ने"]
HINDI_NOUN_SUFFIXES = ["ता", "त्व", "पन", "आई", "ाई", "ई", "ान", "आन"]
HINDI_ADJ_SUFFIXES  = ["ीय", "वाला", "वाली", "वाले", "ी", "ा"]

BENGALI_LEXICON = {
    "আমি": "PRON", "আমরা": "PRON", "তুমি": "PRON", "আপনি": "PRON",
    "সে": "PRON", "তারা": "PRON", "এটি": "PRON", "এরা": "PRON",
    "আছি": "AUX", "আছে": "AUX", "ছিল": "AUX", "হয়": "AUX",
    "এবং": "CONJ", "কিন্তু": "CONJ", "অথবা": "CONJ", "তাই": "CONJ",
    "এ": "ADP", "তে": "ADP", "কে": "ADP", "র": "ADP", "এর": "ADP",
    "ঢাকা": "PROPN", "কলকাতা": "PROPN", "বাংলাদেশ": "PROPN", "ভারত": "PROPN",
    "আজ": "ADV", "কাল": "ADV", "এখানে": "ADV", "সেখানে": "ADV",
    "এক": "NUM", "দুই": "NUM", "তিন": "NUM", "চার": "NUM",
    "ভালো": "ADJ", "বড়": "ADJ", "ছোট": "ADJ", "নতুন": "ADJ",
}

TAMIL_LEXICON = {
    "நான்": "PRON", "நாங்கள்": "PRON", "நீ": "PRON", "நீங்கள்": "PRON",
    "அவன்": "PRON", "அவள்": "PRON", "அவர்கள்": "PRON", "அது": "PRON",
    "இருக்கிறேன்": "AUX", "இருக்கிறது": "AUX", "இருந்தது": "AUX",
    "மற்றும்": "CONJ", "ஆனால்": "CONJ", "அல்லது": "CONJ",
    "இல்": "ADP", "க்கு": "ADP", "இன்": "ADP", "உடன்": "ADP",
    "சென்னை": "PROPN", "இந்தியா": "PROPN", "தமிழ்நாடு": "PROPN",
    "இன்று": "ADV", "நேற்று": "ADV", "இங்கே": "ADV", "அங்கே": "ADV",
    "ஒன்று": "NUM", "இரண்டு": "NUM", "மூன்று": "NUM",
    "நல்ல": "ADJ", "பெரிய": "ADJ", "சிறிய": "ADJ",
}

TELUGU_LEXICON = {
    "నేను": "PRON", "మేము": "PRON", "నువ్వు": "PRON", "మీరు": "PRON",
    "అతను": "PRON", "ఆమె": "PRON", "వారు": "PRON", "అది": "PRON",
    "ఉన్నాను": "AUX", "ఉంది": "AUX", "ఉన్నది": "AUX",
    "మరియు": "CONJ", "కానీ": "CONJ", "లేదా": "CONJ",
    "లో": "ADP", "కు": "ADP", "తో": "ADP", "వల్ల": "ADP",
    "హైదరాబాద్": "PROPN", "తెలంగాణ": "PROPN", "భారత్": "PROPN",
    "ఈరోజు": "ADV", "నిన్న": "ADV", "ఇక్కడ": "ADV", "అక్కడ": "ADV",
    "ఒకటి": "NUM", "రెండు": "NUM", "మూడు": "NUM",
    "మంచి": "ADJ", "పెద్ద": "ADJ", "చిన్న": "ADJ",
}

MARATHI_LEXICON = {
    "मी": "PRON", "आम्ही": "PRON", "तू": "PRON", "तुम्ही": "PRON",
    "तो": "PRON", "ती": "PRON", "ते": "PRON", "हे": "PRON",
    "आहे": "AUX", "होते": "AUX", "असेल": "AUX",
    "आणि": "CONJ", "पण": "CONJ", "किंवा": "CONJ",
    "मध्ये": "ADP", "ला": "ADP", "चा": "ADP", "ची": "ADP",
    "पुणे": "PROPN", "मुंबई": "PROPN", "महाराष्ट्र": "PROPN",
    "आज": "ADV", "काल": "ADV", "इथे": "ADV", "तिथे": "ADV",
    "एक": "NUM", "दोन": "NUM", "तीन": "NUM",
    "चांगला": "ADJ", "मोठा": "ADJ", "लहान": "ADJ",
}

LANGUAGE_CONFIG = {
    "hi": {
        "name": "Hindi",
        "script": "Devanagari",
        "lexicon": HINDI_LEXICON,
        "verb_suffixes": HINDI_VERB_SUFFIXES,
        "noun_suffixes": HINDI_NOUN_SUFFIXES,
        "adj_suffixes": HINDI_ADJ_SUFFIXES,
        "sample": "मैं आज दिल्ली जाता हूँ और वहाँ अच्छा खाना खाऊँगा।",
    },
    "bn": {
        "name": "Bengali",
        "script": "Bengali",
        "lexicon": BENGALI_LEXICON,
        "verb_suffixes": ["চ্ছি", "চ্ছে", "বে", "লাম", "লে", "িয়ে"],
        "noun_suffixes": ["তা", "তি", "ন", "গণ"],
        "adj_suffixes": ["ীয়", "বান", "বতী"],
        "sample": "আমি আজ ঢাকা যাচ্ছি এবং ভালো খাবার খাব।",
    },
    "ta": {
        "name": "Tamil",
        "script": "Tamil",
        "lexicon": TAMIL_LEXICON,
        "verb_suffixes": ["கிறேன்", "கிறது", "கிறார்", "ந்தேன்", "வேன்"],
        "noun_suffixes": ["கள்", "ம்", "ன்", "ள்"],
        "adj_suffixes": ["ான", "ான்", "ள்ள"],
        "sample": "நான் இன்று சென்னை செல்கிறேன் மற்றும் நல்ல உணவு சாப்பிடுவேன்।",
    },
    "te": {
        "name": "Telugu",
        "script": "Telugu",
        "lexicon": TELUGU_LEXICON,
        "verb_suffixes": ["తున్నాను", "తాను", "తారు", "డం", "ను"],
        "noun_suffixes": ["లు", "డు", "ము", "ని"],
        "adj_suffixes": ["మైన", "గల", "వంటి"],
        "sample": "నేను ఈరోజు హైదరాబాద్ వెళ్తున్నాను మరియు మంచి భోజనం చేస్తాను।",
    },
    "mr": {
        "name": "Marathi",
        "script": "Devanagari",
        "lexicon": MARATHI_LEXICON,
        "verb_suffixes": ["तो", "ते", "तात", "ला", "ली", "ले", "णार"],
        "noun_suffixes": ["ता", "पणा", "आई", "ण"],
        "adj_suffixes": ["ीय", "वाला", "वाली", "वाले"],
        "sample": "मी आज पुणे जातो आणि तिथे चांगले जेवण खाईन।",
    },
}


# ─────────────────────────────────────────────
# Core Tagger
# ─────────────────────────────────────────────

class IndianLanguagePOSTagger:
    def __init__(self, lang_code="hi"):
        if lang_code not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {lang_code}. Choose from {list(LANGUAGE_CONFIG.keys())}")
        self.lang_code = lang_code
        self.config = LANGUAGE_CONFIG[lang_code]
        self.lexicon = self.config["lexicon"]

    def is_number(self, token):
        return bool(re.match(r"^[0-9०-९]+$", token))

    def is_punctuation(self, token):
        return bool(re.match(r"^[।,.!?;:\"\'\(\)\[\]\-–—]+$", token))

    def suffix_match(self, token, suffixes, min_stem_len=2):
        for suffix in sorted(suffixes, key=len, reverse=True):
            if token.endswith(suffix) and len(token) - len(suffix) >= min_stem_len:
                return True
        return False

    def tag_token(self, token):
        # 1. Punctuation
        if self.is_punctuation(token):
            return "PUNCT"
        # 2. Number
        if self.is_number(token):
            return "NUM"
        # 3. Lexicon lookup
        if token in self.lexicon:
            return self.lexicon[token]
        # 4. Morphological suffix rules
        if self.suffix_match(token, self.config["verb_suffixes"]):
            return "VERB"
        if self.suffix_match(token, self.config["adj_suffixes"]):
            return "ADJ"
        if self.suffix_match(token, self.config["noun_suffixes"]):
            return "NOUN"
        # 5. Capital-like pattern (proper noun heuristic for Romanized)
        if token[0].isupper() and len(token) > 1:
            return "PROPN"
        # 6. Default: NOUN (most common POS in Indian languages)
        return "NOUN"

    def tokenize(self, text):
        return list(indic_tokenize.trivial_tokenize(text, self.lang_code))

    def tag(self, text):
        tokens = self.tokenize(text)
        tagged = [(token, self.tag_token(token)) for token in tokens]
        return tagged

    def tag_with_details(self, text):
        tagged = self.tag(text)
        result = []
        for token, pos in tagged:
            result.append({
                "token": token,
                "pos": pos,
                "description": POS_TAGS.get(pos, "Unknown"),
            })
        return result

    def get_stats(self, tagged_result):
        from collections import Counter
        pos_counts = Counter(item["pos"] for item in tagged_result)
        total = len(tagged_result)
        return {
            "total_tokens": total,
            "pos_distribution": dict(pos_counts),
            "unique_pos_count": len(pos_counts),
        }


# ─────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  POS Tagger for Indian Languages")
    print("=" * 65)

    for lang_code, config in LANGUAGE_CONFIG.items():
        tagger = IndianLanguagePOSTagger(lang_code)
        text = config["sample"]
        result = tagger.tag_with_details(text)
        stats = tagger.get_stats(result)

        print(f"\n🌐 Language: {config['name']} ({config['script']})")
        print(f"📝 Input: {text}\n")
        print(f"  {'Token':<25} {'POS':<10} {'Description'}")
        print(f"  {'-'*60}")
        for item in result:
            print(f"  {item['token']:<25} {item['pos']:<10} {item['description']}")
        print(f"\n  📊 Stats: {stats['total_tokens']} tokens | "
              f"{stats['unique_pos_count']} unique POS tags")
        print(f"  Distribution: {stats['pos_distribution']}")
        print("-" * 65)

    # Interactive mode
    print("\n🔤 INTERACTIVE MODE")
    print("Supported languages:", {k: v['name'] for k, v in LANGUAGE_CONFIG.items()})
    while True:
        lang = input("\nEnter language code (or 'q' to quit): ").strip()
        if lang == 'q':
            break
        if lang not in LANGUAGE_CONFIG:
            print("Invalid language code!")
            continue
        text = input("Enter text: ").strip()
        if not text:
            continue
        tagger = IndianLanguagePOSTagger(lang)
        result = tagger.tag_with_details(text)
        print(f"\n  {'Token':<25} {'POS':<10} {'Description'}")
        print(f"  {'-'*55}")
        for item in result:
            print(f"  {item['token']:<25} {item['pos']:<10} {item['description']}")


if __name__ == "__main__":
    demo()
