# Corpus of Human–AI Conversations and Linguistic Analysis System

A complete NLP project (built entirely in **Python + Streamlit**) that collects
human ↔ AI conversation pairs, stores them as a structured corpus, and
analyzes/compares their linguistic and sentence structures using **spaCy**
and **NLTK**.

No separate HTML, CSS, or JavaScript files are used — the entire interface
is built with Streamlit.

---

## 1. Features

- **Conversation input** — enter a human message and an AI response, tag it
  with a topic, and save it under an auto-generated unique conversation ID.
- **Corpus storage** — every conversation is stored in a SQLite database
  (`database/corpus.db`) with an automatically synced CSV backup
  (`data/conversations.csv`).
- **NLP preprocessing** — tokenization, sentence segmentation,
  lemmatization, POS tagging, and stop-word analysis (spaCy + NLTK).
- **Sentence-structure analysis** — sentence counts, word counts, unique
  words, lexical diversity, and POS-category counts (nouns, verbs,
  adjectives, adverbs).
- **Human vs AI comparison** — an automatic side-by-side comparison table,
  plus a corpus-wide (average) comparison mode.
- **Interactive visualizations** — Plotly bar/pie charts for word counts,
  sentence length, POS distribution, and lexical diversity.
- **5-page Streamlit dashboard** — Home, Add Conversation, Corpus Dataset,
  Linguistic Analysis, Human vs AI Comparison, all via sidebar navigation.
- **6 sample conversations** ship with the project so you can explore every
  feature immediately without typing anything in.

---

## 2. Project Structure

```
human_ai_corpus/
│
├── app.py                     # Main Streamlit app (UI + navigation)
├── database.py                # SQLite CRUD + CSV sync + sample data
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   └── conversations.csv      # Auto-synced CSV backup of the corpus
│
├── nlp/
│   ├── __init__.py
│   ├── preprocessing.py       # Tokenization, sentence split, lemmas, POS, stop-words
│   ├── analysis.py            # Sentence-structure / linguistic metrics
│   └── comparison.py          # Human vs AI comparison logic
│
└── database/
    └── corpus.db              # SQLite database file (auto-created)
```

---

## 3. Requirements

- Python 3.9+
- pip

---

## 4. Installation

1. **(Recommended) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the spaCy English model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **NLTK data** (punkt, stopwords, POS tagger, wordnet) downloads
   automatically the first time you run the app. If your machine has no
   internet access at that point, download it manually first:

   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4
   ```

---

## 5. Running the App

From the `human_ai_corpus/` folder, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) —
open it in your browser. The database and CSV backup already ship with
**6 sample conversations** covering topics like climate change, Python
programming, healthy living, travel, AI, and cooking, so you can explore
every page immediately.

---

## 6. Using the Dashboard

| Page | What it does |
|---|---|
| 🏠 **Home** | Corpus overview: totals, average word counts, topic breakdown, recent conversations. |
| ➕ **Add Conversation** | Enter a human message + AI response (or generate a quick sample response), tag a topic, and save — a unique conversation ID is suggested automatically. |
| 📚 **Corpus Dataset** | Browse/search/filter all stored conversations, inspect a single one in full, delete conversations, or download the whole corpus as CSV. |
| 🔬 **Linguistic Analysis** | Run full NLP preprocessing (tokens, sentences, lemmas, POS tags, stop-words) and structural metrics on the human message, AI response, or both. |
| ⚖️ **Human vs AI Comparison** | Automatic comparison table + 4 interactive Plotly charts (word count, sentence length, POS distribution, lexical diversity) for a single conversation, or averaged across the whole corpus. |

---

## 7. Notes on the "Generate AI response" feature

The **Add Conversation** page includes an optional *"Generate a simple
sample AI response"* button. This produces a lightweight, template-based
placeholder response for demo purposes — it does **not** call any external
AI API. If you want genuine AI-generated responses, you can extend
`app.py` to call an LLM API of your choice and populate the AI Response
field automatically.

---

## 8. Resetting the Corpus

To start over with an empty corpus, simply delete the database and CSV
files and restart the app — they will be recreated automatically (with
the sample conversations reloaded, since the sample-loader only runs when
the corpus is empty):

```bash
rm database/corpus.db data/conversations.csv
streamlit run app.py
```

---

## 9. Tech Stack

| Purpose | Library |
|---|---|
| UI / Dashboard | Streamlit |
| NLP (tokenize, sentence split, lemmatize, POS tag) | spaCy |
| NLP (stop-words) | NLTK |
| Data handling | Pandas |
| Database | SQLite (via `sqlite3`) |
| Visualization | Plotly |

---

## 10. Troubleshooting

- **`OSError: [E050] Can't find model 'en_core_web_sm'`** — run
  `python -m spacy download en_core_web_sm` again, or manually install:
  `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl`
- **NLTK `LookupError`** — run the manual NLTK download command in
  section 4, step 4.
- **Port already in use** — run `streamlit run app.py --server.port 8502`
  (or any free port).

---

Built as a demonstration project for an NLP / linguistics course, showing
an end-to-end pipeline: data collection → corpus storage → preprocessing →
structural analysis → comparison → visualization, entirely in Python.
