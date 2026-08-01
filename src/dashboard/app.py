"""
Module 2 — Streamlit dashboard.

Run with:
    streamlit run src/dashboard/app.py

Reads the same data/notices_metadata.jsonl file Module 1 already writes
to — no pipeline changes needed, this is purely a read-only view.
"""
import matplotlib.pyplot as plt
import streamlit as st
import os
import sys

# Streamlit sets this script's own folder as the import root, not the
# project root, so "src.notice_store" etc. can't be found without this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import streamlit as st
from src.dashboard.data_loader import (
    load_dashboard_data,
    monthly_counts,
    category_counts,
    authority_counts,
    top_keywords,
)

st.set_page_config(page_title="Academic Notice Dashboard", layout="wide")
st.title("Academic Notice Dashboard")

df = load_dashboard_data()

if df.empty:
    st.warning(
        "No notices found yet. Run `python main.py` on some notice images first, "
        "then reload this page."
    )
    st.stop()

# --- KPI row ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total notices", len(df))
col2.metric("Categories seen", df["category"].nunique())
col3.metric("Issuing authorities", df["issuing_authority"].nunique())

st.divider()

# --- Charts ------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Notices per month")
    monthly = monthly_counts(df)
    if monthly.empty:
        st.info("Not enough notices with a readable date yet to show a trend.")
    else:
        fig, ax = plt.subplots()
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.fill_between(monthly.index, monthly.values, alpha=0.15)
        plt.xticks(rotation=45, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

with right:
    st.subheader("Notices by category")
    cats = category_counts(df)
    fig, ax = plt.subplots()
    ax.pie(cats.values, labels=cats.index, autopct="%1.0f%%")
    st.pyplot(fig)

left2, right2 = st.columns(2)

with left2:
    st.subheader("Notices by issuing authority")
    authorities = authority_counts(df)
    fig, ax = plt.subplots()
    ax.barh(authorities.index, authorities.values, color="#5DCAA5")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

with right2:
    st.subheader("Most frequent subject keywords")
    keywords = top_keywords(df)
    if keywords:
        words, counts = zip(*keywords)
        fig, ax = plt.subplots()
        ax.bar(words, counts, color="#D4537E")
        plt.xticks(rotation=45, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    else:
        st.info("Not enough text yet to extract keywords.")

st.divider()

# --- Filterable raw data table ------------------------------------------
st.subheader("All notices")
selected_category = st.selectbox(
    "Filter by category", ["All"] + sorted(df["category"].unique().tolist())
)
table_df = df if selected_category == "All" else df[df["category"] == selected_category]
st.dataframe(
    table_df[["filename", "date_issued", "issuing_authority", "subject_line", "category"]],
    use_container_width=True,
)