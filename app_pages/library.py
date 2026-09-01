"""The saved papers: filter, set status, open, remove, export."""

import pandas as pd
import streamlit as st

from research_assistant import export, store, ui
from research_assistant.models import STATUSES

st.caption("Everything you saved. Select rows to open, re-label, export or remove them.")

papers = ui.library()
if not papers:
    ui.empty_library()
    st.stop()

counts = {s: sum(p.status == s for p in papers) for s in STATUSES}
with st.container(horizontal=True):
    st.metric("Papers", len(papers), border=True)
    st.metric("To read", counts["To read"], border=True)
    st.metric("Reading", counts["Reading"], border=True)
    st.metric("Read", counts["Read"], border=True)
    st.metric("With full text", sum(p.has_full_text for p in papers), border=True,
              help="Papers whose body was read from arXiv's HTML version. The rest have the abstract only.")

filters = st.columns([3, 2.4, 2], vertical_alignment="bottom")
needle = filters[0].text_input("Filter", placeholder="Title, author or tag", icon=":material/filter_list:",
                               live=True, key="lib_filter")
statuses = filters[1].pills("Status", STATUSES, selection_mode="multi", key="lib_status")
categories = sorted({p.primary_category for p in papers})
category = filters[2].selectbox("Category", [None, *categories], key="lib_category",
                                format_func=lambda c: c or "All categories")

needle = needle.strip().lower()
shown = [p for p in papers
         if (not statuses or p.status in statuses)
         and (not category or p.primary_category == category)
         and (not needle or needle in f"{p.title} {' '.join(p.authors)} {' '.join(p.tags)}".lower())]

frame = pd.DataFrame({
    "Title": [p.title for p in shown],
    "Authors": [p.short_authors for p in shown],
    "Submitted": pd.to_datetime([p.published for p in shown]).date,
    "Category": [p.primary_category for p in shown],
    "Status": [p.status for p in shown],
    "Tags": [p.tags for p in shown],
    "Full text": [p.has_full_text for p in shown],
    "Digest": [bool(p.digests) for p in shown],
    "Citations": [p.citation_count for p in shown],
})
event = st.dataframe(
    frame, hide_index=True, on_select="rerun", selection_mode="multi-row", key="lib_table",
    alt="Saved papers with status, tags and citation counts",
    column_config={
        "Title": st.column_config.TextColumn(width="large"),
        "Submitted": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Tags": st.column_config.ListColumn(),
        "Citations": st.column_config.NumberColumn(help="From Semantic Scholar; update them on the Insights page."),
    })
selected = [shown[i] for i in event.selection.rows if i < len(shown)]


@st.dialog("Remove from library", icon=":material/delete:")
def confirm_removal(ids: list[str], titles: list[str]) -> None:
    st.write(f"Remove {len(ids)} paper{'s' if len(ids) != 1 else ''}? Their digests and your notes are deleted too.")
    for title in titles[:8]:
        st.caption(title)
    if st.button("Remove", type="primary", icon=":material/delete:"):
        for arxiv_id in ids:
            store.delete(arxiv_id)
        st.rerun()


with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(f"{len(selected)} selected" if selected else f"{len(shown)} shown")
    if st.button("Open in reader", icon=":material/menu_book:", type="primary", disabled=len(selected) != 1):
        ui.open_in_reader(selected[0].arxiv_id)
    new_status = st.menu_button("Set status", STATUSES, icon=":material/label:", disabled=not selected)
    if new_status:
        for paper in selected:
            paper.status = new_status
            store.save(paper)
        st.toast(f"Marked {len(selected)} as “{new_status}”.", icon=":material/check_circle:")
        st.rerun()
    st.download_button("Export BibTeX", export.bibtex(selected or shown), file_name="library.bib",
                       mime="text/plain", icon=":material/download:", on_click="ignore",
                       help="The selected papers, or everything shown when nothing is selected.")
    if st.button("Remove", icon=":material/delete:", disabled=not selected):
        confirm_removal([p.arxiv_id for p in selected], [p.title for p in selected])
