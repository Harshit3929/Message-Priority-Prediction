import pandas as pd
import streamlit as st

import database
from nlp.analysis import analyze_text
from nlp.comparison import compare_corpus, compare_texts
from nlp.preprocessing import full_preprocess

st.set_page_config(page_title="Human-AI Corpus", layout="centered")

database.init_db()
database.load_sample_data()


def pick_conversation(label, key=None):
    data = database.get_all_conversations()
    if data.empty:
        st.info("The corpus is empty. Add a conversation first.")
        return None
    options = []
    for row in data.itertuples():
        options.append(f"{row.conversation_id} - {row.topic}")
    choice = st.selectbox(label, options, key=key)
    return choice.split(" - ")[0]


def show_metrics(pairs):
    columns = st.columns(len(pairs))
    position = 0
    for label, value in pairs:
        columns[position].metric(label, value)
        position = position + 1


def side_by_side(table):
    return pd.DataFrame(
        {"Human": table["Human"].tolist(), "AI": table["AI"].tolist()},
        index=table["Metric"].tolist())


st.title("Human-AI Corpus")
st.caption("Human messages paired with AI responses, compared by sentence "
           "structure and word use.")

corpus_tab, analysis_tab = st.tabs(["Corpus", "Analysis"])


with corpus_tab:
    data = database.get_all_conversations()

    if data.empty:
        st.info("The corpus is empty. Add the first conversation below.")
    else:
        show_metrics([
            ("Conversations", len(data)),
            ("Avg human words", round(data["human_word_count"].mean(), 1)),
            ("Avg AI words", round(data["ai_word_count"].mean(), 1)),
            ("Topics", data["topic"].nunique()),
        ])
        st.bar_chart(pd.DataFrame({
            "Human": data["human_word_count"].tolist(),
            "AI": data["ai_word_count"].tolist(),
        }, index=data["conversation_id"].tolist()))

    with st.expander("Add a conversation"):
        with st.form("add_conversation"):
            left, right = st.columns([1, 2])
            conversation_id = left.text_input(
                "ID", value=database.generate_conversation_id())
            topic = right.text_input("Topic", placeholder="e.g. Cooking")
            human_message = st.text_area("Human message", height=110)
            ai_response = st.text_area("AI response", height=140)
            make_sample = st.form_submit_button("Generate a sample AI response")
            saved = st.form_submit_button("Save", type="primary")

        if make_sample:
            if human_message.strip():
                question = human_message.strip().rstrip("?.!")
                st.text_area(
                    "Sample response - copy this into the AI response box",
                    f"That is an interesting question about \"{question}\". "
                    "Generally speaking, this topic involves several factors "
                    "that interact with one another, and understanding it "
                    "fully means looking at both the underlying causes and "
                    "their practical effects. A good next step is to break "
                    "the topic into smaller parts and look at each one on "
                    "its own.",
                    height=110)
            else:
                st.warning("Write a human message first.")

        if saved:
            if not human_message.strip() or not ai_response.strip():
                st.error("Both the human message and AI response are required.")
            else:
                try:
                    database.add_conversation(conversation_id, topic,
                                              human_message, ai_response)
                    st.success(f"Saved {conversation_id}.")
                    st.rerun()
                except ValueError as problem:
                    st.error(str(problem))

    if not data.empty:
        st.divider()
        left, right = st.columns([1, 2])
        topics = ["All"] + sorted(data["topic"].unique().tolist())
        chosen_topic = left.selectbox("Topic", topics)
        search = right.text_input("Search", placeholder="keyword")

        shown = data
        if chosen_topic != "All":
            shown = shown[shown["topic"] == chosen_topic]
        if search.strip():
            in_human = shown["human_message"].str.contains(
                search, case=False, na=False)
            in_ai = shown["ai_response"].str.contains(
                search, case=False, na=False)
            shown = shown[in_human | in_ai]

        st.dataframe(
            shown[["conversation_id", "topic", "timestamp",
                   "human_word_count", "ai_word_count"]],
            width="stretch", hide_index=True)
        st.caption(f"{len(shown)} of {len(data)} conversations")

        conversation_id = pick_conversation("Open a conversation", "open")
        if conversation_id:
            record = database.get_conversation_by_id(conversation_id)
            if record is not None:
                human_side, ai_side = st.columns(2)
                with human_side:
                    st.markdown("**Human**")
                    st.write(record["human_message"])
                    st.caption(f"{record['human_word_count']} words")
                with ai_side:
                    st.markdown("**AI**")
                    st.write(record["ai_response"])
                    st.caption(f"{record['ai_word_count']} words")
                if st.button("Delete"):
                    database.delete_conversation(conversation_id)
                    st.success(f"Deleted {conversation_id}.")
                    st.rerun()

        st.download_button("Download CSV",
                           data.to_csv(index=False).encode("utf-8"),
                           file_name="conversations.csv", mime="text/csv")


with analysis_tab:
    scope = st.radio("Scope", ["One conversation", "Whole corpus"],
                     horizontal=True)

    if scope == "Whole corpus":
        table = compare_corpus(database.get_all_conversations())
        if table is None:
            st.info("The corpus is empty. Add a conversation first.")
        else:
            st.caption("Averages across every conversation.")
            st.dataframe(table, width="stretch", hide_index=True)
            st.bar_chart(side_by_side(table))
    else:
        conversation_id = pick_conversation("Conversation", "analyse")
        if conversation_id:
            record = database.get_conversation_by_id(conversation_id)

            table, human, ai = compare_texts(record["human_message"],
                                             record["ai_response"])
            st.dataframe(table, width="stretch", hide_index=True)
            st.bar_chart(side_by_side(table))

            st.divider()
            side = st.radio("Break down", ["Human", "AI", "Both"],
                            horizontal=True)

            sides = []
            if side in ("Human", "Both"):
                sides.append(("Human", record["human_message"]))
            if side in ("AI", "Both"):
                sides.append(("AI", record["ai_response"]))

            for name, text in sides:
                st.subheader(name)
                st.write(text)

                steps = full_preprocess(text)
                numbers = analyze_text(text)

                show_metrics([
                    ("Sentences", numbers["total_sentences"]),
                    ("Words", numbers["total_words"]),
                    ("Unique", numbers["unique_words"]),
                    ("Avg length", numbers["avg_sentence_length"]),
                    ("Diversity", numbers["lexical_diversity"]),
                ])
                show_metrics([
                    ("Nouns", numbers["noun_count"]),
                    ("Verbs", numbers["verb_count"]),
                    ("Adjectives", numbers["adjective_count"]),
                    ("Adverbs", numbers["adverb_count"]),
                ])

                if numbers["pos_distribution"]:
                    st.bar_chart(pd.Series(numbers["pos_distribution"]))

                with st.expander(f"Tokens and sentences - {name}"):
                    st.caption(f"{len(steps['words'])} tokens")
                    st.write(steps["words"])
                    number = 1
                    for sentence in steps["sentences"]:
                        st.write(f"{number}. {sentence}")
                        number = number + 1

                with st.expander(f"Lemmas - {name}"):
                    st.dataframe(
                        pd.DataFrame(steps["lemmas"],
                                     columns=["Token", "Lemma"]),
                        width="stretch", hide_index=True)

                with st.expander(f"POS tags - {name}"):
                    st.dataframe(
                        pd.DataFrame(steps["pos_tags"],
                                     columns=["Token", "POS", "Tag"]),
                        width="stretch", hide_index=True)

                with st.expander(f"Stop words - {name}"):
                    found = steps["stopwords"]
                    show_metrics([
                        ("Stop words", found["stopword_count"]),
                        ("Other words", found["non_stopword_count"]),
                        ("Ratio", found["stopword_ratio"]),
                    ])
                    st.write(", ".join(found["stopwords_found"]) or "None")
