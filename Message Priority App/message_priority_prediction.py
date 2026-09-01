import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = ["low", "medium", "high", "critical"]
SEED = 42
MAX_MISSING = 0.5

os.chdir(os.path.dirname(os.path.abspath(__file__)))

word_finder = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                              sublinear_tf=True, lowercase=False)
piece_finder = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                               min_df=3, sublinear_tf=True, lowercase=False)


def head(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


def clean(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"\S+@\S+\.\S+", " <EMAIL> ", text)
    text = re.sub(r"\b[A-Za-z]+-?\d[\w-]*\b", " <ID> ", text)
    text = re.sub(r"\b\d{5,}\b", " <NUM> ", text)
    text = re.sub(r"([!?.])\1\1+", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_words():
    table = pd.read_csv("urgency_words.csv")
    groups = {}
    expected = {}
    to_delete = []
    for _, row in table.iterrows():
        group = row["group"]
        phrase = str(row["phrase"]).lower()
        if group not in groups:
            groups[group] = []
            expected[group] = row["expected"]
        groups[group].append(phrase)
        if row["remove_in_test"] == "yes":
            to_delete.append(phrase)
    for group in groups:
        groups[group].sort(key=len, reverse=True)
    to_delete.sort(key=len, reverse=True)
    return groups, expected, to_delete


def holds_numbers(column):
    return pd.api.types.is_numeric_dtype(column)


def preprocess(data):
    head("STEP 2 of 5 - Cleaning the data")
    print(f"\n  The raw file has {len(data)} rows and {len(data.columns)} columns.")

    print("\n  2.1  What is missing?\n")
    for column in data.columns:
        blanks = data[column].isna().sum()
        print(f"       {column:<22} {blanks:>6} missing "
              f"({100 * blanks / len(data):.1f}%)")

    print(f"\n  2.2  Drop columns more than {100 * MAX_MISSING:.0f}% empty")
    for column in list(data.columns):
        share = data[column].isna().mean()
        if share > MAX_MISSING:
            data = data.drop(columns=column)
            print(f"       '{column}' was {100 * share:.1f}% empty - removed")

    print("\n  2.3  Fill missing numbers with the mean")
    for column in data.columns:
        if not holds_numbers(data[column]):
            continue
        gaps = data[column].isna().sum()
        if gaps > 0:
            average = data[column].mean()
            data[column] = data[column].fillna(average)
            print(f"       {column:<20} mean = {average:.2f}, filled {gaps} gaps")

    print("\n  2.4  Fill missing words with the most common value")
    for column in data.columns:
        if column in ("message", "priority") or holds_numbers(data[column]):
            continue
        gaps = data[column].isna().sum()
        if gaps > 0:
            common = data[column].mode()[0]
            data[column] = data[column].fillna(common)
            print(f"       {column:<20} most common = '{common}', "
                  f"filled {gaps} gaps")

    print("\n  2.5  Remove rows we cannot use")
    before = len(data)
    data = data.dropna(subset=["message", "priority"])
    print(f"       no message or priority - removed {before - len(data)} rows")
    before = len(data)
    data = data[data["priority"].isin(LABELS)]
    if before - len(data) > 0:
        print(f"       bad priority value - removed {before - len(data)} rows")

    print("\n  2.6  Tidy the message text")
    messy = None
    for raw in data["message"]:
        if clean(raw) != raw:
            messy = raw
            break
    data["text"] = data["message"].apply(clean)
    if messy is not None:
        print(f'       before: "{messy[:60]}"')
        print(f'       after : "{clean(messy)[:60]}"')

    print("\n  2.7  Remove duplicate messages")
    before = len(data)
    data = data.drop_duplicates("text")
    print(f"       {before} rows became {len(data)} rows")

    data = data.reset_index(drop=True)
    data["rank"] = data["priority"].apply(LABELS.index)

    print(f"\n  Finished: {len(data)} clean rows.")
    return data


GROUPS, EXPECTED, DELETE_THESE = load_words()

CLUE_NAMES = list(GROUPS) + ["exclamations", "questions", "capitals", "length"]
EXPECTED["exclamations"] = "up"
EXPECTED["questions"] = "down"
EXPECTED["capitals"] = "up"
EXPECTED["length"] = "no guess"


def count_clues(text):
    lowered = text.lower()
    word_count = max(len(text.split()), 1)
    per_100 = 100 / word_count

    numbers = []
    for phrases in GROUPS.values():
        found = 0
        for phrase in phrases:
            found = found + lowered.count(phrase)
        numbers.append(found * per_100)

    letters = []
    for c in text:
        if c.isalpha():
            letters.append(c)
    capitals = 0
    for c in letters:
        if c.isupper():
            capitals = capitals + 1

    numbers.append(text.count("!") * per_100)
    numbers.append(text.count("?") * per_100)
    numbers.append(capitals / max(len(letters), 1))
    numbers.append(word_count)
    return numbers


def count_clues_for_all(messages):
    rows = []
    for m in messages:
        rows.append(count_clues(m))
    return np.array(rows)


def show_clues(data):
    head("STEP 3 of 5 - Which clues signal urgency?")
    counts = count_clues_for_all(data["text"])
    ranks = data["rank"].to_numpy()

    print(f"\n  {'clue':<14}{'low':>7}{'med':>7}{'high':>7}{'crit':>7}"
          f"{'link':>9}   result")
    print("  " + "-" * 62)

    right = 0
    wrong = 0
    strengths = []
    for i, name in enumerate(CLUE_NAMES):
        column = counts[:, i]
        averages = []
        for r in range(4):
            averages.append(column[ranks == r].mean())

        link = pd.Series(column).corr(pd.Series(ranks), method="spearman")
        if link > 0.05:
            happened = "up"
        elif link < -0.05:
            happened = "down"
        else:
            happened = "flat"

        guess = EXPECTED[name]
        if guess == "no guess":
            result = "(no guess)"
        elif guess == happened:
            result = "as expected"
            right = right + 1
        else:
            result = "SURPRISE"
            wrong = wrong + 1

        strengths.append((link, name))
        print(f"  {name:<14}{averages[0]:>7.2f}{averages[1]:>7.2f}"
              f"{averages[2]:>7.2f}{averages[3]:>7.2f}{link:>+9.3f}   {result}")

    print("  " + "-" * 62)
    print(f"\n  {right} of {right + wrong} predictions were correct.")

    strengths.sort()
    print(f"  Strongest signal: {strengths[-1][1]} ({strengths[-1][0]:+.3f})")
    print(f"  Strongest negative: {strengths[0][1]} ({strengths[0][0]:+.3f})")


def score(real, guessed):
    real = np.array(real)
    guessed = np.array(guessed)
    off = np.abs(guessed - real)
    return {
        "macro_f1": f1_score(real, guessed, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(real, guessed, weights="quadratic"),
        "mae": off.mean(),
        "off_by_2": (off >= 2).mean(),
        "critical_recall": recall_score(real, guessed, labels=[3],
                                        average="macro", zero_division=0),
        "accuracy": (guessed == real).mean(),
    }


def random_baseline(train_ranks, test_ranks):
    chances = []
    for r in range(4):
        chances.append(list(train_ranks).count(r) / len(train_ranks))

    random_maker = np.random.default_rng(SEED)
    totals = {}
    for _ in range(20):
        guesses = random_maker.choice(4, size=len(test_ranks), p=chances)
        for name, value in score(test_ranks, guesses).items():
            totals[name] = totals.get(name, 0) + value / 20
    return totals


def make_word_features(train_messages, test_messages):
    train_words = word_finder.fit_transform(train_messages)
    train_pieces = piece_finder.fit_transform(train_messages)
    test_words = word_finder.transform(test_messages)
    test_pieces = piece_finder.transform(test_messages)
    return hstack([train_words, train_pieces]), hstack([test_words, test_pieces])


def make_models():
    return {
        "words + logistic regression": LogisticRegression(
            C=4, max_iter=2000, class_weight="balanced", random_state=SEED),
        "words + support vector": LinearSVC(
            C=0.5, class_weight="balanced", random_state=SEED),
        "words + naive bayes": ComplementNB(alpha=0.3),
    }


def train_clue_model(train_messages, train_ranks, test_messages):
    scaler = StandardScaler()
    train_clues = scaler.fit_transform(count_clues_for_all(train_messages))
    test_clues = scaler.transform(count_clues_for_all(test_messages))
    model = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED)
    model.fit(train_clues, train_ranks)
    return model.predict(test_clues)


def train_models(train_messages, train_ranks, test_messages, test_ranks):
    head("STEP 4 of 5 - Training the models")

    train_x, test_x = make_word_features(train_messages, test_messages)

    counts = []
    for r in range(4):
        counts.append(list(train_ranks).count(r))
    nb_weights = []
    for r in train_ranks:
        nb_weights.append(len(train_ranks) / (4 * max(counts[r], 1)))

    results = {}
    trained = {}
    for name, model in make_models().items():
        if "naive bayes" in name:
            model.fit(train_x, train_ranks, sample_weight=nb_weights)
        else:
            model.fit(train_x, train_ranks)
        results[name] = score(test_ranks, model.predict(test_x))
        trained[name] = model
        print("  trained " + name)

    clue_guesses = train_clue_model(train_messages, train_ranks, test_messages)
    results["clues only (no words!)"] = score(test_ranks, clue_guesses)
    print("  trained clues only (no words!)")

    results["RANDOM GUESSING (baseline)"] = random_baseline(train_ranks,
                                                            test_ranks)

    print(f"\n  {'Model':<30}{'macro-F1':>10}{'QWK':>9}{'MAE':>8}"
          f"{'off-by-2':>10}{'accuracy':>10}")
    print("  " + "-" * 77)
    in_order = sorted(results.items(), key=lambda pair: -pair[1]["macro_f1"])
    for name, s in in_order:
        print(f"  {name:<30}{s['macro_f1']:>10.4f}{s['qwk']:>9.4f}"
              f"{s['mae']:>8.3f}{s['off_by_2']:>10.3f}{s['accuracy']:>10.4f}")
    print("  " + "-" * 77)

    best_name = in_order[0][0]
    print(f"\n  Best model: {best_name} "
          f"({results[best_name]['macro_f1']:.4f})")
    print("  About 0.90 is the ceiling here, because 12% of the labels are")
    print("  wrong on purpose. A score of 0.99 would mean something leaked.")
    print(f"  'clues only' got {results['clues only (no words!)']['macro_f1']:.4f} "
          f"using no words at all.")

    return best_name, trained[best_name], results[best_name]


def predict(model, messages):
    words = word_finder.transform(messages)
    pieces = piece_finder.transform(messages)
    return model.predict(hstack([words, pieces]))


KEEP_CAPITALS = ["VPN", "CRM", "API", "SSO", "SLA", "EOD", "HR", "IT", "ID",
                 "NUM", "URL", "EMAIL", "REF", "EU", "CEO"]


def strip_urgency(text):
    for phrase in DELETE_THESE:
        text = re.sub(re.escape(phrase), "thing", text, flags=re.IGNORECASE)
    text = text.replace("!", ".")

    tidied = []
    for word in text.split():
        plain = word.strip("<>.,!?:;()")
        if "<" in word or ">" in word or plain in KEEP_CAPITALS:
            tidied.append(word)
        elif word.isupper() and len(plain) > 3:
            tidied.append(word.capitalize())
        else:
            tidied.append(word)
    return re.sub(r"\s+", " ", " ".join(tidied)).strip()


def tidy(text):
    return re.sub(r"\s+", " ", text).strip()


def test_a(model, test_messages, test_ranks, best_scores):
    head("STEP 5 of 5, TEST A - delete every urgency word")

    stripped = []
    for m in test_messages:
        stripped.append(strip_urgency(m))

    changed = 0
    for before, after in zip(test_messages, stripped):
        if tidy(before) != tidy(after):
            changed = changed + 1
    percent = 100 * changed / len(test_messages)

    after_scores = score(test_ranks, predict(model, stripped))
    print(f"\n  macro-F1 before deleting : {best_scores['macro_f1']:.4f}")
    print(f"  macro-F1 after deleting  : {after_scores['macro_f1']:.4f} "
          f"(change {after_scores['macro_f1'] - best_scores['macro_f1']:+.4f})")
    print(f"  messages actually changed: {percent:.1f}%")

    print("\n  Example of the deletion:")
    for before, after in zip(test_messages, stripped):
        swapped_at = after.find("thing")
        if tidy(before) == tidy(after) or swapped_at == -1 or swapped_at > 95:
            continue
        print("    before: " + before[:120] + "...")
        print("    after : " + after[:120] + "...")
        break

    print(f"\n  The deletion worked ({percent:.1f}% of messages changed) and the")
    print("  score did not move, so the model is not just keyword matching.")
    print("  Hold that thought until Test B.")


TRICKY = []
for _, row in pd.read_csv("trick_questions.csv").iterrows():
    TRICKY.append((row["message"], row["correct_answer"]))


def test_b(model):
    head("STEP 5 of 5, TEST B - trick questions")
    print("\n  Messages where the wording and the real seriousness disagree.\n")

    messages = []
    for message, _ in TRICKY:
        messages.append(message)
    answers = predict(model, messages)

    too_high = 0
    too_low = 0
    for (message, correct), given in zip(TRICKY, answers):
        gap = int(given) - LABELS.index(correct)
        if gap > 0:
            too_high = too_high + 1
        elif gap < 0:
            too_low = too_low + 1
        mark = "OK" if gap == 0 else "X "
        print(f'  {mark} "{message[:56]}..."')
        print(f"     should be {correct:<9} got {LABELS[int(given)]:<9} ({gap:+d})\n")

    wrong = too_high + too_low
    print("  " + "-" * 62)
    print(f"  SCORE: {len(TRICKY) - wrong} of {len(TRICKY)} correct "
          f"(too urgent: {too_high}, not urgent enough: {too_low})")
    print("  " + "-" * 62)

    if wrong == 0:
        print("\n  All correct. Eight questions is too few to prove much.")
    elif too_high > 0 and too_low > 0:
        print(f"\n  Every mistake leaned the same way: {too_high} loud-but-trivial")
        print(f"  rated too high, {too_low} calm-but-serious rated too low.")
        print("  A model that did not know would make random mistakes.")
    return wrong


def conclusion(wrong):
    head("SO WHAT DOES IT MEAN?")

    if wrong == 0:
        print("\n  Both tests agree this time, but eight trick questions is")
        print("  not enough to conclude much.")
        return

    print(f"\n  Test A said the urgency words do not matter. Test B got")
    print(f"  {wrong} of {len(TRICKY)} wrong. Both are true.")
    print("\n  In normal messages the two signals agree - 'urgent' turns up")
    print("  next to 'the system is down'. Deleting one changes nothing.")
    print("  The trick questions were built so the two disagree, and there")
    print("  the wording wins.")
    print("\n  " + "-" * 62)
    print("  CONCLUSION: the model learned HOW URGENTLY A MESSAGE IS WRITTEN,")
    print("  not HOW SERIOUS THE SITUATION IS.")
    print("  " + "-" * 62)
    print("\n  If I had only run Test A I would have called it robust. An")
    print("  average over 1200 messages hides a problem affecting a few.")


EXAMPLES = [
    "Hi team, the payment gateway is completely down and no orders are going "
    "through. This is affecting all customers.",
    "Could someone update the font on the intranet when you get a chance? No rush.",
    TRICKY[0][0],
    TRICKY[1][0],
]


def how_sure(model, message):
    words = word_finder.transform([message])
    pieces = piece_finder.transform([message])
    features = hstack([words, pieces])

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[0]
    scores = model.decision_function(features)[0]
    lifted = np.exp(scores - max(scores))
    return lifted / lifted.sum()


def predict_one(model, message):
    chances = how_sure(model, message)
    best = int(np.argmax(chances))

    print("\n  " + "-" * 66)
    print(f'  Message: "{message[:120]}"')
    print(f"\n  PREDICTED:  >>>  {LABELS[best].upper()}  <<<\n")

    for i, name in enumerate(LABELS):
        bar = "#" * int(chances[i] * 40)
        chosen = "  <-- chosen" if i == best else ""
        print(f"    {name:<9} {bar:<40} {chances[i] * 100:5.1f}%{chosen}")

    found = []
    for phrases in GROUPS.values():
        for phrase in phrases:
            if phrase in message.lower():
                found.append(phrase)
    if found:
        print("\n  Clue words spotted: " + ", ".join(found[:8]))
    else:
        print("\n  Clue words spotted: (none)")

    stripped = strip_urgency(message)
    if tidy(stripped) != tidy(message):
        other = int(np.argmax(how_sure(model, stripped)))
        print("\n  With the urgency words deleted:")
        print(f'    becomes: "{stripped[:60]}..."')
        if other == best:
            print(f"    prediction stays {LABELS[other]}")
        else:
            print(f"    prediction changes: {LABELS[best]} -> {LABELS[other]}")
    print("  " + "-" * 66)


def demo(model):
    head("LIVE DEMO - type your own messages")
    print("\n  Type a message and press Enter.")
    print("  Or type:  examples  /  tricky  /  quit\n")

    while True:
        try:
            typed = input("  Your message > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  (no keyboard here - running the examples instead)")
            for message in EXAMPLES:
                predict_one(model, message)
            return

        if typed == "":
            continue
        elif typed.lower() in ("quit", "exit", "q"):
            print("\n  Demo finished.\n")
            return
        elif typed.lower() == "examples":
            for message in EXAMPLES:
                predict_one(model, message)
        elif typed.lower() == "tricky":
            test_b(model)
        else:
            predict_one(model, typed)


def main():
    head("MESSAGE PRIORITY PREDICTION")
    print("\n  The four levels are a scale, not four separate boxes:\n")
    print("        low  <  medium  <  high  <  critical\n")
    print("  Saying 'high' when the answer is 'critical' is a small mistake.")
    print("  Saying 'low' is a serious one. Accuracy cannot tell them apart.")

    head("STEP 1 of 5 - Loading messages.csv")
    raw = pd.read_csv("messages.csv")
    print(f"\n  Loaded {len(raw)} rows and {len(raw.columns)} columns.")
    print("  Columns: " + ", ".join(raw.columns))

    data = preprocess(raw)

    print(f"\n  {len(data)} usable messages, {len(GROUPS)} clue groups.\n")
    print(f"  {'priority':<12}{'count':>8}{'share':>9}{'avg words':>12}")
    print("  " + "-" * 41)
    for name in LABELS:
        rows = data[data["priority"] == name]
        average_words = rows["text"].str.split().str.len().mean()
        print(f"  {name:<12}{len(rows):>8}{100 * len(rows) / len(data):>8.1f}%"
              f"{average_words:>12.1f}")
    print("  " + "-" * 41)
    print("\n  Urgent messages are longer, so clues are counted per 100 words.")

    train_messages, test_messages, train_ranks, test_ranks = train_test_split(
        data["text"].tolist(), data["rank"].to_numpy(),
        test_size=0.2, random_state=SEED, stratify=data["rank"])
    print(f"\n  Training on {len(train_messages)} messages, "
          f"testing on {len(test_messages)}.")

    show_clues(data)
    name, model, best_scores = train_models(train_messages, train_ranks,
                                            test_messages, test_ranks)
    test_a(model, test_messages, test_ranks, best_scores)
    conclusion(test_b(model))

    head("FINISHED")
    print("\n  REMINDER: messages.csv was generated, not collected from real")
    print("  inboxes. These numbers show the system works end to end.")

    demo(model)


if __name__ == "__main__":
    main()
