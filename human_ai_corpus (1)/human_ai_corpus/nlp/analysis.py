from nlp.preprocessing import clean_text, read

POS_LABELS = {
    "NOUN": "Nouns", "PROPN": "Proper Nouns", "VERB": "Verbs",
    "AUX": "Auxiliary Verbs", "ADJ": "Adjectives", "ADV": "Adverbs",
    "PRON": "Pronouns", "DET": "Determiners", "ADP": "Prepositions",
    "CCONJ": "Coordinating Conjunctions", "SCONJ": "Subordinating Conjunctions",
    "NUM": "Numerals", "PART": "Particles", "INTJ": "Interjections",
    "PUNCT": "Punctuation", "SYM": "Symbols", "X": "Other",
}

EMPTY = {
    "total_sentences": 0, "total_words": 0, "unique_words": 0,
    "avg_sentence_length": 0.0, "lexical_diversity": 0.0,
    "noun_count": 0, "verb_count": 0, "adjective_count": 0,
    "adverb_count": 0, "pos_distribution": {},
}


def analyze_text(text):
    text = clean_text(text)
    if not text:
        return dict(EMPTY)

    doc = read(text)

    sentences = []
    for sentence in doc.sents:
        if sentence.text.strip():
            sentences.append(sentence)

    words = []
    for token in doc:
        if token.is_alpha:
            words.append(token)

    total_sentences = len(sentences)
    total_words = len(words)

    seen = set()
    for token in words:
        seen.add(token.text.lower())
    unique_words = len(seen)

    counts = {}
    for token in words:
        counts[token.pos_] = counts.get(token.pos_, 0) + 1

    distribution = {}
    for tag in sorted(counts):
        distribution[POS_LABELS.get(tag, tag)] = counts[tag]

    avg_length = round(total_words / total_sentences, 2) if total_sentences else 0.0
    diversity = round(unique_words / total_words, 3) if total_words else 0.0

    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_sentence_length": avg_length,
        "lexical_diversity": diversity,
        "noun_count": counts.get("NOUN", 0) + counts.get("PROPN", 0),
        "verb_count": counts.get("VERB", 0) + counts.get("AUX", 0),
        "adjective_count": counts.get("ADJ", 0),
        "adverb_count": counts.get("ADV", 0),
        "pos_distribution": distribution,
    }
