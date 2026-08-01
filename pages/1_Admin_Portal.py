"""
Admin Portal — upload notice images here. Each upload is saved to
data/input_docs, then automatically run through the same pipeline as
`python main.py`: extraction -> classification -> storage -> chunking ->
embedding -> vector database. No separate step needed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from config import INPUT_PATH
from main import process_new_notices
from src.ui_common import get_retriever
from src.notice_store import load_all_notices

st.set_page_config(page_title="Admin Portal", page_icon="📤", layout="centered")

st.title("📤 Admin Portal")
st.caption("Upload notice images. They'll be extracted, classified, and made searchable automatically.")

uploaded_files = st.file_uploader(
    "Upload notice images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and st.button(f"Process {len(uploaded_files)} file(s)", type="primary"):
    saved_names = []
    for f in uploaded_files:
        dest_path = os.path.join(INPUT_PATH, f.name)
        with open(dest_path, "wb") as out_file:
            out_file.write(f.getbuffer())
        saved_names.append(f.name)

    with st.status("Processing uploaded notices...", expanded=True) as status:
        st.write(f"Saved {len(saved_names)} file(s) to `data/input_docs/`.")

        st.write("Extracting + classifying (calls Gemini, may take a moment per notice)...")
        documents_to_add = process_new_notices()
        st.write(f"Extraction complete — {len(documents_to_add)} searchable chunks created.")

        st.write("Embedding and storing in the vector database...")
        get_retriever(new_documents=documents_to_add)

        status.update(label="Done. These notices are now searchable on the Chat page.", state="complete")

    st.success(f"Processed {len(saved_names)} notice(s): {', '.join(saved_names)}")

st.divider()

st.subheader("Recently processed notices")
all_notices = load_all_notices()
if not all_notices:
    st.info("No notices processed yet.")
else:
    recent = list(reversed(all_notices))[:10]
    st.dataframe(
        [
            {
                "filename": n["filename"],
                "category": n.get("category"),
                "subject_line": (n.get("subject_line") or "")[:80],
            }
            for n in recent
        ],
        use_container_width=True,
    )