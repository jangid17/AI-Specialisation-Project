"""Read one paper: digest, questions, full text, notes."""

import streamlit as st

from research_assistant import analysis, export, fulltext, llm, retrieval, store, ui
from research_assistant.models import STATUSES

st.caption("A digest you can check against the text, answers with the passages they came from, and your notes.")

papers = ui.library()
if not papers:
    ui.empty_library()
    st.stop()

by_id = {p.arxiv_id: p for p in papers}
state = st.session_state
if "paper" in state and state["paper"] not in by_id:  # the paper was removed from the library
    state["paper"] = papers[0].arxiv_id
paper_id = st.selectbox("Paper", list(by_id), format_func=lambda i: ui.paper_label(by_id[i]), key="paper",
                        bind="query-params", label_visibility="collapsed")
paper = by_id[paper_id]

st.header(paper.title, anchor=False)
st.markdown(", ".join(paper.authors[:15]) + (f" and {len(paper.authors) - 15} more" if len(paper.authors) > 15 else ""))
with st.container(horizontal=True, vertical_alignment="center"):
    st.badge(paper.primary_category, color="blue")
    st.badge(f"Submitted {paper.published}", icon=":material/event:", color="gray")
    if paper.has_full_text:
        words = len(paper.full_text().split())
        st.badge(f"Full text · {words:,} words", icon=":material/article:", color="green")
    else:
        st.badge("Abstract only", icon=":material/warning:", color="orange")
    if paper.reference_count:
        st.badge(f"{paper.reference_count} references", color="gray")
    if paper.citation_count is not None:
        st.badge(f"{paper.citation_count:,} citations", icon=":material/format_quote:", color="gray")
    st.link_button("arXiv", paper.abs_url, icon=":material/open_in_new:", type="tertiary")
    st.link_button("PDF", paper.pdf_url, icon=":material/picture_as_pdf:", type="tertiary")

meta = st.columns([1.3, 2], vertical_alignment="bottom")
status = meta[0].segmented_control("Reading status", STATUSES, default=paper.status, required=True,
                                   key=f"status_{paper_id}")
all_tags = sorted({t for p in papers for t in p.tags})
tags = meta[1].multiselect("Tags", all_tags, default=paper.tags, accept_new_options=True,
                           placeholder="Add a tag", key=f"tags_{paper_id}")
if status != paper.status or tags != paper.tags:
    paper.status, paper.tags = status, tags
    store.save(paper)

VIEWS = {"Digest": ":material/summarize: Digest", "Ask": ":material/forum: Ask",
         "Full text": ":material/article: Full text", "Notes": ":material/edit_note: Notes"}
view = st.segmented_control("View", list(VIEWS), default="Digest", required=True, key="reader_view",
                            format_func=VIEWS.get, label_visibility="collapsed")


def show_passages(passages: list[dict]) -> None:
    with st.expander(f"Passages used ({len(passages)})", icon=":material/format_quote:"):
        for i, p in enumerate(passages, start=1):
            st.markdown(f"**[{i}] {p['title']} — {p['section']}**")
            st.caption(p["text"].replace("## ", ""))


if view == "Digest":
    controls = st.columns([1.2, 2.2, 1.2], vertical_alignment="bottom")
    depth = controls[0].segmented_control(
        "Depth", analysis.DEPTHS, default="Focused", required=True, key="digest_depth",
        disabled=not paper.has_full_text,
        help="Focused: one model call over the abstract, introduction, conclusion and the passages closest "
             "to method, results and limitations. Full paper: notes on every section first, then the digest.")
    audience = controls[1].segmented_control("Written for", list(analysis.AUDIENCES), default="Specialist",
                                             required=True, key="digest_audience")
    key = analysis.digest_key(depth, audience)
    digest = paper.digests.get(key)
    generate = controls[2].button("Regenerate" if digest else "Generate digest", icon=":material/auto_awesome:",
                                  type="secondary" if digest else "primary", width="stretch", key="generate_digest",
                                  disabled=not llm.is_configured())
    if depth == "Full paper" and not digest:
        calls = analysis.full_paper_calls(paper)
        st.caption(f"Reads the whole paper in {calls} model calls. On Groq's free tier (8,000 tokens a minute) "
                   f"expect roughly {max(1, calls // 2)} minute{'s' if calls > 3 else ''}.")
    ui.llm_ready()

    if generate:
        try:
            with st.status("Reading the paper", expanded=True) as progress_box:
                bar = st.progress(0.0, text="Starting")
                digest = analysis.make_digest(
                    paper, depth, audience, on_wait=ui.notify_wait,
                    on_progress=lambda i, n: bar.progress(i / n, text=f"Step {i + 1} of {n}"))
                bar.progress(1.0, text="Done")
                progress_box.update(label="Digest ready", state="complete", expanded=False)
            paper.digests[key] = digest
            store.save(paper)
            st.rerun()
        except llm.LLMError as e:
            st.error(str(e), icon=":material/error:")

    if digest:
        results = digest["key_results"]
        unverified = sum(not r.get("verified") for r in results)
        st.info(digest["tldr"], title="In brief", icon=":material/bolt:")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Problem", anchor=False)
            st.write(digest["problem"])
            st.subheader("Method", anchor=False)
            st.write(digest["method"])
        with right:
            st.subheader("Contributions", anchor=False)
            st.markdown("\n".join(f"- {c}" for c in digest["contributions"]))
            st.subheader("Background needed", anchor=False)
            st.markdown("\n".join(f"- {c}" for c in digest["prerequisites"]))

        st.subheader("Key results", anchor=False)
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r['claim']}**")
                st.markdown(f"> {r['evidence']}" if r["evidence"] else "*No supporting quote was given.*")
                if r.get("verified"):
                    st.badge("Quote found in the paper", icon=":material/verified:", color="green")
                else:
                    st.badge("Quote not found word for word", icon=":material/help:", color="orange",
                             help="The model's supporting quote does not appear in the extracted text. "
                                  "It may be paraphrased, or come from a table or formula. Check the paper.")

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Limitations", anchor=False)
            st.markdown("\n".join(f"- {c}" for c in digest["limitations"]))
        with right:
            st.subheader("Questions for discussion", anchor=False)
            st.markdown("\n".join(f"1. {c}" for c in digest["discussion_questions"]))
        if digest["who_should_read"]:
            st.markdown(f":material/group: **Who should read it:** {digest['who_should_read']}")

        with st.container(horizontal=True, vertical_alignment="center"):
            st.download_button("Download as Markdown", export.digest_markdown(paper, digest),
                               file_name=f"{export.bibtex_key(paper)}.md", mime="text/markdown",
                               icon=":material/download:", on_click="ignore")
            st.caption(f"Written by {digest.get('model', 'the model')} from the {digest.get('basis', 'paper')}. "
                       + (f"{unverified} of {len(results)} supporting quotes could not be matched to the text."
                          if unverified else "Every supporting quote was matched to the text."))
    elif not generate:
        st.caption("No digest for this depth and audience yet.")

elif view == "Ask":
    scope = st.segmented_control("Answer from", ["This paper", "Whole library"], default="This paper",
                                 required=True, key="ask_scope")
    whole = scope == "Whole library"
    history = state.setdefault("chat_library" if whole else f"chat_{paper_id}", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("passages"):
                show_passages(message["passages"])

    question = st.chat_input("Ask about your library" if whole else "Ask about this paper",
                             submit_mode="disable", disabled=not llm.is_configured(), key="ask_input")
    SUGGESTIONS = ["What is the main contribution?", "How was the method evaluated?",
                   "What do the authors say are the limitations?", "What assumptions does the method rely on?"]
    if not history:
        chips = st.empty()
        picked = chips.pills("Suggestions", SUGGESTIONS, label_visibility="collapsed", key=f"chips_{scope}_{paper_id}")
        question = question or picked
        if question:
            chips.empty()
    ui.llm_ready()

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status(":shimmer[Finding the relevant passages]", type="compact") as finding:
                previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                found = retrieval.search(f"{previous} {question}".strip(),
                                         list(by_id) if whole else [paper_id], top_k=8 if whole else 6)
                finding.update(label=f"Read {len(found)} passages", state="complete")
            passages = [{"title": by_id[p.arxiv_id].title, "section": p.section, "text": p.text} for p in found]
            try:
                answer = st.write_stream(analysis.answer_stream(
                    question, found, {i: p.title for i, p in by_id.items()},
                    [{"role": m["role"], "content": m["content"]} for m in history], on_wait=ui.notify_wait))
                show_passages(passages)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer, "passages": passages})
            except llm.LLMError as e:
                st.error(str(e), icon=":material/error:")
    if history and st.button("Clear conversation", icon=":material/delete_sweep:", type="tertiary"):
        history.clear()
        st.rerun()

elif view == "Full text":
    if not paper.has_full_text:
        st.warning("arXiv had no HTML version of this paper when it was saved, so only the abstract is stored.",
                   icon=":material/warning:")
        st.write(paper.abstract)
        if st.button("Try again", icon=":material/refresh:"):
            try:
                with st.spinner("Fetching the full text", show_time=True):
                    fulltext.fetch_full_text(paper)
                    store.save(paper)
                    retrieval.build_index(paper)
                st.rerun()
            except fulltext.FullTextUnavailable as e:
                st.error(str(e), icon=":material/error:")
    else:
        st.caption(f"Extracted from [{paper.fulltext_source}]({paper.fulltext_source}). Tables and figures are "
                   "left out; their captions are kept in square brackets.")
        with st.expander("Abstract", expanded=True):
            st.write(paper.abstract)
        for i, section in enumerate(paper.sections):
            with st.expander(section.title, key=f"sec_{paper_id}_{i}"):
                st.markdown(section.text.replace("\n## ", "\n##### ").removeprefix("## "))

else:
    def save_notes() -> None:
        paper.notes = state[f"notes_{paper_id}"]
        store.save(paper)

    st.text_area("Notes", value=paper.notes, height=320, key=f"notes_{paper_id}", on_change=save_notes,
                 placeholder="Your reading notes. Saved when you click outside the box. "
                             "They are included in the Markdown export of the digest.")
    st.caption("Saved automatically." if paper.notes else "Nothing saved yet.")
