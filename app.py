import streamlit as st
import config
from pipeline import run as run_pipeline, update_page
from search import search

st.set_page_config(page_title="MF Guide Search", layout="centered")
st.title("Fannie Mae Multifamily Guide — Semantic Search")

# ── Pipeline ──────────────────────────────────────────────────────────────────

with st.expander("Admin"):
    if st.button("Run ingestion pipeline", type="secondary"):
        with st.spinner("Scraping, embedding, and indexing all sections..."):
            try:
                run_pipeline()
                st.success("Pipeline complete.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    st.divider()
    st.markdown("**Update a single page**")
    update_url = st.text_input("Page URL", placeholder="https://mfguide.fanniemae.com/node/1234", key="update_url")
    if st.button("Update page", type="secondary", disabled=not update_url):
        with st.spinner(f"Updating {update_url}..."):
            try:
                update_page(update_url)
                st.success(f"Page updated successfully.")
            except Exception as e:
                st.error(f"Update failed: {e}")

# ── Search ────────────────────────────────────────────────────────────────────

query = st.text_input("Search the guide", placeholder="e.g. delegating underwriting authority")

if query:
    with st.spinner("Searching..."):
        try:
            hits = search(query, k=5)
        except Exception as e:
            st.error(f"Search failed: {e}")
            hits = []

    if not hits:
        st.info("No results found.")
    else:
        for hit in hits:
            src     = hit["_source"]
            score   = hit["_score"]
            title   = src.get("page_title", "Untitled")
            section = src.get("section_heading", "")
            url     = src.get("source_url", "")
            text    = src.get("text", "")
            pos     = f"{src.get('chunk_index', 0) + 1}/{src.get('total_chunks', '?')}"

            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
                    if section:
                        st.caption(f"Section: {section}  ·  chunk {pos}")
                with col2:
                    st.metric("Score", f"{score:.3f}")
                st.write(text)
