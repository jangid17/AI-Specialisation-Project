"""Compare saved papers and draft a related-work paragraph."""

import pandas as pd
import streamlit as st

from research_assistant import analysis, export, llm, ui

st.caption("Put two to five papers side by side and get a paragraph you can start a literature review from.")

papers = ui.library()
if len(papers) < 2:
    st.info("Save at least two papers to compare them.", title="Not enough papers yet",
            icon=":material/compare_arrows:")
    st.page_link(ui.DISCOVER_PAGE, label="Find papers", icon=":material/travel_explore:")
    st.stop()

by_id = {p.arxiv_id: p for p in papers}
state = st.session_state

controls = st.columns([5, 1], vertical_alignment="bottom")
chosen = controls[0].multiselect("Papers to compare", list(by_id), max_selections=5, key="compare_ids",
                                 format_func=lambda i: ui.paper_label(by_id[i]), placeholder="Choose two to five")
run = controls[1].button("Compare", type="primary", icon=":material/compare_arrows:", width="stretch", key="run_compare",
                         disabled=len(chosen) < 2 or not llm.is_configured())
ui.llm_ready()
with_digest = sum(bool(by_id[i].digests) for i in chosen)
if chosen:
    st.caption(f"{with_digest} of {len(chosen)} selected papers have a digest. Papers with a digest are compared "
               "on their method, results and limitations; the others on their abstract only.")

if run:
    try:
        with st.spinner("Comparing the papers", show_time=True):
            state.comparison = {"ids": list(chosen),
                                "data": analysis.compare([by_id[i] for i in chosen], on_wait=ui.notify_wait)}
    except llm.LLMError as e:
        st.error(str(e), icon=":material/error:")

comparison = state.get("comparison")
if comparison and all(i in by_id for i in comparison["ids"]):
    compared = [by_id[i] for i in comparison["ids"]]
    data = comparison["data"]
    rows = [r for r in ui.as_list(data.get("rows")) if isinstance(r, dict)]
    table = pd.DataFrame([{
        "Paper": f"[{n}] {p.short_authors} ({p.year})",
        "Problem": r.get("problem", ""), "Approach": r.get("approach", ""), "Evidence": r.get("evidence", ""),
        "Key result": r.get("key_result", ""), "Limitation": r.get("limitation", ""),
    } for n, (p, r) in enumerate(zip(compared, rows), start=1)])
    st.subheader("Side by side", anchor=False)
    st.table(table, hide_index=True, alt="Problem, approach, evidence, key result and limitation of each paper")

    columns = st.columns(3, gap="medium")
    for column, title, icon, field in [
            (columns[0], "Where they agree", ":material/handshake:", "agreements"),
            (columns[1], "Where they differ", ":material/call_split:", "differences"),
            (columns[2], "Open questions", ":material/help:", "open_questions")]:
        with column.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.markdown("\n".join(f"- {item}" for item in ui.as_list(data.get(field))) or "Nothing reported.")

    st.subheader("Related-work draft", anchor=False)
    references = "\n".join(f"[{n}] {', '.join(p.authors[:6])}{' et al' if len(p.authors) > 6 else ''}. "
                           f"{p.title}. arXiv:{p.arxiv_id}, {p.year}." for n, p in enumerate(compared, start=1))
    with st.container(border=True):
        st.write(data.get("synthesis", ""))
        st.caption(references.replace("\n", "  \n"))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button("Download draft", f"{data.get('synthesis', '')}\n\n{references}\n",
                           file_name="related-work.txt", mime="text/plain", icon=":material/download:",
                           on_click="ignore")
        st.download_button("BibTeX for these papers", export.bibtex(compared), file_name="compared.bib",
                           mime="text/plain", icon=":material/download:", on_click="ignore")
        st.caption("A draft written by the model from the digests and abstracts. Check each claim against "
                   "the papers before using it.")
