import nltk
import spacy

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

try:
    nlp_model = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_model = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    if text is None:
        return ""
    return str(text).strip()


def read(text):
    return nlp_model(clean_text(text))


def tokenize_words(text):
    words = []
    for token in read(text):
        if not token.is_space:
            words.append(token.text)
    return words


def tokenize_sentences(text):
    sentences = []
    for sentence in read(text).sents:
        if sentence.text.strip():
            sentences.append(sentence.text.strip())
    return sentences


def lemmatize_text(text):
    pairs = []
    for token in read(text):
        if not token.is_space:
            pairs.append((token.text, token.lemma_))
    return pairs


def pos_tagging(text):
    tags = []
    for token in read(text):
        if not token.is_space:
            tags.append((token.text, token.pos_, token.tag_))
    return tags


def stopword_analysis(text):
    words = []
    for word in tokenize_words(text):
        if word.isalpha():
            words.append(word.lower())

    found = []
    for word in words:
        if word in STOP_WORDS:
            found.append(word)

    total = len(words)
    ratio = round(len(found) / total, 3) if total else 0.0

    return {
        "total_words": total,
        "stopword_count": len(found),
        "non_stopword_count": total - len(found),
        "stopword_ratio": ratio,
        "stopwords_found": sorted(set(found)),
    }


def full_preprocess(text):
    return {
        "words": tokenize_words(text),
        "sentences": tokenize_sentences(text),
        "lemmas": lemmatize_text(text),
        "pos_tags": pos_tagging(text),
        "stopwords": stopword_analysis(text),
    }
