import pandas as pd

from nlp.analysis import analyze_text

METRIC_LABELS = [
    ("total_words", "Word Count"),
    ("total_sentences", "Sentence Count"),
    ("avg_sentence_length", "Average Sentence Length"),
    ("unique_words", "Unique Words"),
    ("lexical_diversity", "Lexical Diversity"),
    ("noun_count", "Noun Count"),
    ("verb_count", "Verb Count"),
    ("adjective_count", "Adjective Count"),
    ("adverb_count", "Adverb Count"),
]


def compare_texts(human_text, ai_text):
    human = analyze_text(human_text)
    ai = analyze_text(ai_text)

    rows = []
    for key, label in METRIC_LABELS:
        rows.append({
            "Metric": label,
            "Human": human[key],
            "AI": ai[key],
            "Difference": round(ai[key] - human[key], 3),
        })

    return pd.DataFrame(rows), human, ai


def compare_corpus(conversations):
    if conversations is None or conversations.empty:
        return None

    human_results = []
    ai_results = []
    for _, row in conversations.iterrows():
        human_results.append(analyze_text(row["human_message"]))
        ai_results.append(analyze_text(row["ai_response"]))

    total = len(conversations)
    rows = []
    for key, label in METRIC_LABELS:
        human_sum = 0
        ai_sum = 0
        for result in human_results:
            human_sum = human_sum + result[key]
        for result in ai_results:
            ai_sum = ai_sum + result[key]
        human_avg = round(human_sum / total, 3)
        ai_avg = round(ai_sum / total, 3)
        rows.append({
            "Metric": label,
            "Human": human_avg,
            "AI": ai_avg,
            "Difference": round(ai_avg - human_avg, 3),
        })

    return pd.DataFrame(rows)
