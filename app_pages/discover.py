"""Find papers: search arXiv, follow a watchlist, or add by identifier."""

from datetime import date, timedelta

import streamlit as st

from research_assistant import analysis, arxiv_client as ax, llm, store, ui
from research_assistant.models import Paper

st.caption("Search arXiv, follow topics, and see new work ranked against what you care about.")

profile = store.load_profile()
state = st.session_state


def category_label(code: str | None) -> str:
    return "Any category" if code is None else f"{code} · {ax.CATEGORIES[code]}"


def watch_label(watch: dict) -> str:
    parts = [watch["terms"]] if watch["terms"] else []
    if watch.get("category"):
        parts.append(watch["category"])
    return " · ".join(parts)


def save_paper(paper: Paper) -> None:
    with st.spinner(f"Fetching full text of “{paper.title[:60]}”", show_time=True):
        note = ui.add_to_library(paper)
    st.toast(note or "Saved to your library with full text.",
             icon=":material/warning:" if note else ":material/check_circle:")


def paper_card(paper: Paper, key: str, score: float | None = None, brief: dict | None = None) -> None:
    with st.container(border=True):
        body, actions = st.columns([5, 1.2])
        with body:
            st.markdown(f"**[{paper.title}]({paper.abs_url})**")
            st.caption(f"{paper.short_authors} · {paper.published} · {paper.primary_category}")
            if brief:
                colour = {"high": "green", "medium": "orange"}.get(str(brief.get("priority", "")).lower(), "gray")
                st.markdown(f":{colour}-badge[{str(brief.get('priority', '?')).capitalize()} priority] "
                            f"{brief.get('why', '')}")
            with st.expander("Abstract"):
                st.write(paper.abstract)
        with actions:
            if score is not None:
                st.caption(f"Similarity {score:.2f}",
                           help="Cosine similarity between this abstract and your interests statement, "
                                "from the local embedding model. Useful for ordering, not as a percentage.")
            if store.contains(paper.arxiv_id):
                st.badge("In library", icon=":material/check:", color="green")
                if st.button("Open in reader", key=f"open_{key}", icon=":material/menu_book:", width="stretch"):
                    ui.open_in_reader(paper.arxiv_id)
            elif st.button("Save", key=f"add_{key}", type="primary", icon=":material/bookmark_add:",
                           width="stretch"):
                save_paper(paper)
                st.rerun()


def ranked(papers: list[Paper], by_interest: bool) -> list[tuple[Paper, float | None]]:
    if not (by_interest and profile["interests"].strip() and papers):
        return [(p, None) for p in papers]
    scores = ui.interest_scores(tuple(p.abstract for p in papers), profile["interests"])
    return sorted(zip(papers, scores), key=lambda pair: -pair[1])


with st.expander("My research interests", icon=":material/person_search:",
                 expanded=not profile["interests"].strip()):
    interests = st.text_area(
        "Describe what you work on, in a sentence or two",
        value=profile["interests"], height=80,
        placeholder="e.g. Evaluation of retrieval-augmented generation, faithfulness of LLM citations, "
                    "and information retrieval for scientific literature.",
        help="Search results and new papers are ordered by how close their abstract is to this statement.",
        key="interests_text")
    if st.button("Save interests", icon=":material/save:", key="save_interests") and interests != profile["interests"]:
        profile["interests"] = interests
        store.save_profile(profile)
        st.toast("Interests saved.", icon=":material/check_circle:")
        st.rerun()

view = st.segmented_control(
    "View", ["Search", "Watchlist", "Add by ID"], default="Search", required=True,
    label_visibility="collapsed", key="discover_view",
    format_func=lambda v: {"Search": ":material/search: Search arXiv",
                           "Watchlist": ":material/notifications: Watchlist",
                           "Add by ID": ":material/link: Add by ID or URL"}[v])

if view == "Search":
    with st.form("search_form", border=False):
        cols = st.columns([4, 1.3, 2.2, 1.6, 1.1], vertical_alignment="bottom")
        terms = cols[0].text_input("Search terms", placeholder='"retrieval augmented generation" evaluation',
                                   help="Words are combined with AND. Put a phrase in double quotes.", key="q_terms")
        field = cols[1].selectbox("Search in", list(ax.SEARCH_FIELDS), key="q_field")
        category = cols[2].selectbox("Category", [None, *ax.CATEGORIES], format_func=category_label, key="q_category")
        sort = cols[3].selectbox("Sort by", list(ax.SORT_OPTIONS), key="q_sort")
        submitted = cols[4].form_submit_button("Search", type="primary", icon=":material/search:", width="stretch")

    if submitted:
        try:
            with st.spinner("Searching arXiv", show_time=True):
                found, total = ax.search(terms, ax.SEARCH_FIELDS[field], category, ax.SORT_OPTIONS[sort],
                                         max_results=25)
            state.search = {"papers": [p.to_dict() for p in found], "total": total,
                            "watch": {"terms": terms.strip(), "field": ax.SEARCH_FIELDS[field],
                                      "category": category}}
        except ax.ArxivError as e:
            st.error(str(e), icon=":material/error:")

    if "search" in state:
        result = state.search
        papers = [Paper.from_dict(d) for d in result["papers"]]
        head = st.container(horizontal=True, vertical_alignment="center")
        head.markdown(f"**{len(papers)}** of {result['total']:,} matches")
        by_interest = head.toggle("Order by my interests", value=bool(profile["interests"].strip()),
                                  disabled=not profile["interests"].strip(), key="search_by_interest")
        if result["watch"] not in profile["watchlist"]:
            if head.button("Follow this search", icon=":material/notification_add:", key="follow_search",
                           help="Adds it to the watchlist so you can check for new papers later."):
                profile["watchlist"].append(result["watch"])
                store.save_profile(profile)
                st.toast("Added to your watchlist.", icon=":material/check_circle:")
                st.rerun()
        else:
            head.badge("On your watchlist", icon=":material/notifications_active:", color="blue")
        if not papers:
            st.info("No papers match. Try fewer words or a different field.", icon=":material/search_off:")
        for paper, score in ranked(papers, by_interest):
            paper_card(paper, f"s_{paper.arxiv_id}", score)

elif view == "Watchlist":
    if not profile["watchlist"]:
        st.info("Run a search and choose “Follow this search” to add it here.",
                title="Nothing on your watchlist yet", icon=":material/notifications:")
    else:
        for i, watch in enumerate(profile["watchlist"]):
            row = st.container(horizontal=True, vertical_alignment="center")
            row.markdown(f":material/notifications: **{watch_label(watch)}**")
            if row.button("Remove", key=f"unwatch_{i}", icon=":material/close:", type="tertiary"):
                profile["watchlist"].pop(i)
                store.save_profile(profile)
                st.rerun()

        controls = st.container(horizontal=True, vertical_alignment="bottom")
        days = controls.segmented_control("Submitted in the last", [7, 14, 30], default=7, required=True,
                                          format_func=lambda d: f"{d} days", key="watch_days")
        if controls.button("Check for new papers", type="primary", icon=":material/refresh:", key="check_new"):
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            fresh: dict[str, Paper] = {}
            try:
                progress = st.progress(0.0, text="Asking arXiv")
                for i, watch in enumerate(profile["watchlist"]):
                    progress.progress(i / len(profile["watchlist"]), text=f"Checking “{watch_label(watch)}”")
                    found, _ = ax.search(watch["terms"], watch["field"], watch["category"],
                                         "submittedDate", max_results=50)
                    fresh.update({p.arxiv_id: p for p in found if p.published >= cutoff})
                progress.empty()
                state.watch_result = {"papers": [p.to_dict() for p in fresh.values()], "days": days}
                state.pop("watch_brief", None)
            except ax.ArxivError as e:
                st.error(str(e), icon=":material/error:")

        if "watch_result" in state:
            papers = [Paper.from_dict(d) for d in state.watch_result["papers"]]
            pairs = ranked(papers, by_interest=True)
            head = st.container(horizontal=True, vertical_alignment="center")
            head.markdown(f"**{len(papers)}** new papers in the last {state.watch_result['days']} days")
            top = [p for p, _ in pairs[:8]]
            if top and ui.llm_ready() and head.button(
                    f"Brief me on the top {len(top)}", icon=":material/auto_awesome:", key="brief_me",
                    help="One model call that reads the abstracts and says which papers deserve your time."):
                try:
                    with st.spinner("Reading the abstracts", show_time=True):
                        brief = analysis.brief_new_papers(top, profile["interests"], on_wait=ui.notify_wait)
                    items = {top[int(item["n"]) - 1].arxiv_id: item for item in ui.as_list(brief.get("items"))
                             if str(item.get("n", "")).isdigit() and 0 < int(item["n"]) <= len(top)}
                    state.watch_brief = {"overview": brief.get("overview", ""), "items": items}
                except llm.LLMError as e:
                    st.error(str(e), icon=":material/error:")
            brief = state.get("watch_brief")
            if brief and brief["overview"]:
                st.info(brief["overview"], title="This batch in brief", icon=":material/auto_awesome:")
            if not papers:
                st.info("Nothing new in that window.", icon=":material/inbox:")
            for paper, score in pairs:
                paper_card(paper, f"w_{paper.arxiv_id}", score, (brief or {}).get("items", {}).get(paper.arxiv_id))

else:
    with st.form("add_form", border=False):
        raw = st.text_area("arXiv IDs or URLs", height=100,
                           placeholder="1706.03762\nhttps://arxiv.org/abs/2405.07437",
                           help="One per line, or separated by commas or spaces.")
        add = st.form_submit_button("Add to library", type="primary", icon=":material/bookmark_add:")
    if add:
        ids = list(dict.fromkeys(i for i in (ax.parse_arxiv_id(t) for t in raw.replace(",", " ").split()) if i))
        if not ids:
            st.error("No arXiv identifier found in that text.", icon=":material/error:")
        else:
            try:
                with st.spinner("Looking the papers up on arXiv", show_time=True):
                    found = ax.fetch_by_ids(ids)
                for paper in found:
                    save_paper(paper)
                missing = set(ids) - {p.arxiv_id for p in found}
                if missing:
                    st.warning(f"arXiv has no paper with ID {', '.join(sorted(missing))}.", icon=":material/warning:")
                if found:
                    st.success(f"Added {len(found)} paper{'s' if len(found) != 1 else ''}.",
                               icon=":material/check_circle:")
                    st.page_link("app_pages/library.py", label="Go to the library", icon=":material/library_books:")
            except ax.ArxivError as e:
                st.error(str(e), icon=":material/error:")
