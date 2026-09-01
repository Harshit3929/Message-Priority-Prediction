import contextlib
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

import message_priority_prediction as mp

st.set_page_config(page_title="Message Priority Prediction",
                   page_icon="📥", layout="centered")

LEVEL_COLOURS = ["#BFD3E6", "#7FA8C9", "#E8A33D", "#C0392B"]

PICKS = {
    "Emergency": "URGENT: the payment gateway is completely down and no "
                 "orders are going through. Please fix this immediately, it "
                 "is affecting all customers.",
    "Minor": "Could someone update the font on the intranet when you get a "
             "chance? No rush.",
    "Loud but trivial": mp.TRICKY[0][0],
    "Calm but serious": mp.TRICKY[1][0],
}


@st.cache_resource(show_spinner="Loading the data and training the models...")
def prepare():
    raw = pd.read_csv("messages.csv")

    with contextlib.redirect_stdout(io.StringIO()):
        data = mp.preprocess(raw.copy())

    train_x, test_x, train_y, test_y = train_test_split(
        data["text"].tolist(), data["rank"].to_numpy(),
        test_size=0.2, random_state=mp.SEED, stratify=data["rank"])

    train_features, test_features = mp.make_word_features(train_x, test_x)

    counts = []
    for r in range(4):
        counts.append(list(train_y).count(r))
    nb_weights = []
    for r in train_y:
        nb_weights.append(len(train_y) / (4 * max(counts[r], 1)))

    results = {}
    trained = {}
    for name, model in mp.make_models().items():
        if "naive bayes" in name:
            model.fit(train_features, train_y, sample_weight=nb_weights)
        else:
            model.fit(train_features, train_y)
        results[name] = mp.score(test_y, model.predict(test_features))
        trained[name] = model

    results["RANDOM GUESSING"] = mp.random_baseline(train_y, test_y)

    best_name = ""
    best_f1 = -1
    for name in trained:
        if results[name]["macro_f1"] > best_f1:
            best_f1 = results[name]["macro_f1"]
            best_name = name

    return {"n_messages": len(data), "results": results,
            "model": trained[best_name], "best": results[best_name]}


ready = prepare()
model = ready["model"]
best = ready["best"]
random_f1 = ready["results"]["RANDOM GUESSING"]["macro_f1"]


st.title("Message Priority Prediction")
st.caption("How urgently does this message need handling?  "
           "`low  <  medium  <  high  <  critical`")

a, b, c = st.columns(3)
a.metric("Messages", f"{ready['n_messages']:,}")
b.metric("Best macro-F1", f"{best['macro_f1']:.2f}")
c.metric("Random guessing", f"{random_f1:.2f}")

st.divider()

if "message" not in st.session_state:
    st.session_state.message = PICKS["Emergency"]

columns = st.columns(len(PICKS))
i = 0
for label in PICKS:
    if columns[i].button(label, width="stretch"):
        st.session_state.message = PICKS[label]
    i = i + 1

message = st.text_area("Message", key="message", height=120)

if message.strip():
    chances = mp.how_sure(model, message)
    winner = int(np.argmax(chances))

    left, right = st.columns(2)

    with left:
        colour = ["blue", "blue", "orange", "red"][winner]
        st.markdown(f"## :{colour}[{mp.LABELS[winner].upper()}]")

        bars = pd.DataFrame({"Level": mp.LABELS, "Chance": chances})
        chart = alt.Chart(bars).mark_bar().encode(
            x=alt.X("Chance:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
            y=alt.Y("Level:N", sort=mp.LABELS, title=None),
            color=alt.Color("Level:N", sort=mp.LABELS, legend=None,
                            scale=alt.Scale(domain=mp.LABELS,
                                            range=LEVEL_COLOURS)),
            tooltip=["Level", alt.Tooltip("Chance:Q", format=".1%")],
        ).properties(height=150)
        st.altair_chart(chart, width="stretch")

    with right:
        st.markdown("**With the urgency words deleted**")
        stripped = mp.strip_urgency(message)
        if mp.tidy(stripped) == mp.tidy(message):
            st.info("No urgency words to remove, so nothing changes.")
        else:
            other = int(np.argmax(mp.how_sure(model, stripped)))
            if other == winner:
                st.success(f"Still **{mp.LABELS[other]}** - the model was not "
                           f"relying on those words here.")
            else:
                st.error(f"Changes: **{mp.LABELS[winner]}** to "
                         f"**{mp.LABELS[other]}** - the model was leaning on "
                         f"the wording.")
            st.text_area("After deleting", stripped, height=120,
                         disabled=True, key="stripped_box")

st.divider()
st.caption("The model learned **how urgently a message is written**, not how "
           "serious the situation is - try the two trick examples above. "
           "The data is generated, not real, so these numbers show the system "
           "works rather than real-world accuracy.")
