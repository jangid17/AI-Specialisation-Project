"""Charts computed from the library. No number on this page comes from the language model."""

import math

import altair as alt
import pandas as pd
import streamlit as st

from research_assistant import citations, metrics, store, ui

st.caption("What your library looks like. Every figure is computed from the saved arXiv records.")

papers = ui.library()
if not papers:
    ui.empty_library()
    st.stop()

graph = metrics.coauthor_graph(papers)
network = metrics.network_stats(graph)
dates = sorted(p.published for p in papers)

with st.container(horizontal=True):
    st.metric("Papers", len(papers), border=True)
    st.metric("Authors", network["authors"], border=True)
    st.metric("Categories", len({p.primary_category for p in papers}), border=True)
    st.metric("Earliest", dates[0], border=True)
    st.metric("Latest", dates[-1], border=True)

BAR = {"cornerRadiusEnd": 4}
COUNT_AXIS = alt.Axis(tickMinStep=1, format="d")
# Chart heights below include the axis, and label space is reserved explicitly: the browser measures
# label widths before the web font has loaded, which otherwise clips long labels on the left.
LABELS = alt.Axis(minExtent=150, labelLimit=140)

left, right = st.columns(2, gap="medium")
with left.container(border=True, height="stretch"):
    timeline, unit = metrics.submissions_over_time(papers)
    st.markdown(f"**Papers by {unit} of first submission**")
    st.altair_chart(
        alt.Chart(timeline).mark_bar(**BAR).encode(
            x=alt.X("period:O", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y("papers:Q", title="Papers", axis=COUNT_AXIS),
            tooltip=[alt.Tooltip("period:O", title=unit.capitalize()), alt.Tooltip("papers:Q", title="Papers")],
        ).properties(height=260),
        alt=f"Number of saved papers per {unit} of first arXiv submission")
    st.caption("Date of the first arXiv version. Periods with no papers are shown as empty.")

with right.container(border=True, height="stretch"):
    by_category = metrics.category_counts(papers)
    st.markdown("**Papers by primary arXiv category**")
    st.altair_chart(
        alt.Chart(by_category).mark_bar(**BAR).encode(
            y=alt.Y("category:N", title=None, sort=None, axis=LABELS),
            x=alt.X("papers:Q", title="Papers", axis=COUNT_AXIS),
            tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("papers:Q", title="Papers")],
        ).properties(height=30 * len(by_category) + 70),
        alt="Number of saved papers in each primary arXiv category")
    st.caption("Each paper is counted once, under the category its authors chose as primary.")

left, right = st.columns(2, gap="medium")
with left.container(border=True, height="stretch"):
    st.markdown("**Characteristic terms**")
    terms = metrics.distinctive_terms(papers)
    if terms.empty:
        st.caption("Needs at least two papers that share a term.")
    else:
        st.altair_chart(
            alt.Chart(terms).mark_bar(**BAR).encode(
                y=alt.Y("term:N", title=None, sort=None, axis=LABELS),
                x=alt.X("papers:Q", title="Papers containing the term", axis=COUNT_AXIS,
                        scale=alt.Scale(domain=[0, len(papers)])),
                tooltip=[alt.Tooltip("term:N", title="Term"), alt.Tooltip("papers:Q", title="Papers")],
            ).properties(height=26 * len(terms) + 70),
            alt="Terms that characterise the library and how many papers contain each")
        st.caption(f"The {len(terms)} words and two-word phrases with the highest average TF-IDF weight in titles "
                   f"and abstracts. Bar length is the number of papers containing the term, out of {len(papers)}.")

with right.container(border=True, height="stretch"):
    st.markdown("**Topic map**")
    topics = None
    if len(papers) >= metrics.MIN_PAPERS_FOR_TOPICS:
        topics = metrics.topic_map(papers, ui.abstract_vectors(tuple(f"{p.title}. {p.abstract}" for p in papers)))
    if topics is None:
        st.caption(f"Needs at least {metrics.MIN_PAPERS_FOR_TOPICS} papers; the library has {len(papers)}.")
    else:
        st.altair_chart(
            alt.Chart(topics["frame"]).mark_point(filled=True, size=120, opacity=0.9).encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                color=alt.Color("topic:N", title="Group (most characteristic terms)",
                                legend=alt.Legend(orient="bottom", columns=1, labelLimit=500, titleLimit=500)),
                shape=alt.Shape("topic:N", legend=None),
                tooltip=[alt.Tooltip("title:N", title="Paper"), alt.Tooltip("authors:N", title="Authors"),
                         alt.Tooltip("year:Q", title="Year", format="d"), alt.Tooltip("topic:N", title="Group")],
            ).properties(height=300),
            alt=f"Saved papers placed by abstract similarity, in {topics['topics']} groups")
        st.caption(
            f"Each point is a paper; nearby points have similar abstracts. Groups are k-means clusters of the "
            f"abstract embeddings, with k chosen by silhouette score ({topics['silhouette']:.2f}; below about "
            f"0.25 the groups overlap heavily). The two axes are a PCA projection that keeps "
            f"{topics['variance_kept']:.0%} of the variance, so distances are approximate and the axes have no units.")
        with st.expander("Papers in each group"):
            st.dataframe(topics["frame"][["topic", "title", "authors", "year"]].sort_values("topic"),
                         hide_index=True, alt="Papers listed by topic group",
                         column_config={"year": st.column_config.NumberColumn(format="%d")})

with st.container(border=True):
    st.markdown("**Co-author network**")
    stats = st.container(horizontal=True)
    stats.metric("Authors", network["authors"])
    stats.metric("Co-author pairs", network["collaborations"])
    stats.metric("Density", f"{network['density']:.3f}",
                 help="Co-author pairs divided by all possible pairs of authors.")
    stats.metric("Separate groups", network["groups"],
                 help="Connected components: sets of authors linked by at least one joint paper.")
    stats.metric("In several papers", network["recurring_authors"],
                 help="Authors who appear on more than one saved paper.")

    drawn = graph
    note = ""
    if graph.number_of_nodes() > 150:
        hubs = {a for a, d in graph.nodes(data=True) if d["papers"] > 1}
        keep = hubs | {n for h in hubs for n in graph.neighbors(h)}
        if len(keep) > 250 or not keep:
            keep = hubs or set(sorted(graph, key=graph.degree, reverse=True)[:150])
        drawn = graph.subgraph(keep)
        note = (f" Showing {drawn.number_of_nodes()} of {graph.number_of_nodes()} authors: those on more than "
                "one paper and their co-authors. The figures above cover the whole network.")
    st.echarts_chart(
        {
            "tooltip": {"formatter": "{b}"},
            "series": [{
                "type": "graph", "layout": "force", "roam": True, "draggable": True,
                "force": {"repulsion": 70, "edgeLength": 45, "gravity": 0.2},
                "label": {"position": "right", "fontSize": 11},
                "labelLayout": {"hideOverlap": True},
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"opacity": 0.45},
                "data": [{"name": a, "value": d["papers"], "symbolSize": 7 + 7 * math.sqrt(d["papers"]),
                          "label": {"show": d["papers"] > 1 or drawn.number_of_nodes() <= 40}}
                         for a, d in drawn.nodes(data=True)],
                "links": [{"source": a, "target": b, "lineStyle": {"width": min(1 + d["weight"], 6)}}
                          for a, b, d in drawn.edges(data=True)],
            }],
        },
        height=520, alt=f"Co-author network of {drawn.number_of_nodes()} authors in the library")
    st.caption("A line joins two authors who share a saved paper; thicker lines mean more joint papers, larger "
               "dots mean more saved papers. Names are matched exactly as arXiv lists them, so namesakes merge "
               "and spelling variants split." + note)

with st.container(border=True):
    head = st.container(horizontal=True, vertical_alignment="center")
    head.markdown("**Citation counts**")
    if head.button("Update from Semantic Scholar", icon=":material/refresh:", key="update_citations"):
        try:
            with st.spinner("Asking Semantic Scholar", show_time=True):
                found = citations.update_citation_counts(papers)
            for paper in papers:
                store.save(paper)
            st.toast(f"Found citation counts for {found} of {len(papers)} papers.", icon=":material/check_circle:")
            st.rerun()
        except citations.CitationsUnavailable as e:
            st.error(str(e), icon=":material/error:")
    cited = [p for p in papers if p.citation_count is not None]
    if not cited:
        st.caption("No citation counts yet. arXiv does not provide them; this fetches them from Semantic Scholar.")
    else:
        top = sorted(cited, key=lambda p: -p.citation_count)[:15]
        frame = pd.DataFrame({
            # Short labels; the full title is in the tooltip. The arXiv id keeps two papers by the same
            # first author in the same year apart.
            "paper": [f"{p.authors[0].split()[-1]}{' et al.' if len(p.authors) > 1 else ''} {p.year} · {p.arxiv_id}"
                      for p in top],
            "title": [p.title for p in top],
            "citations": [p.citation_count for p in top],
        })
        # Citation counts are long-tailed. When the largest is over 50 times the smallest, linear bars would
        # reduce all but one paper to a sliver, so the chart switches to dots on a logarithmic axis (bars
        # must start at zero, which a log axis cannot do).
        skewed = frame["citations"].max() > 50 * max(frame["citations"].min(), 1)
        base = alt.Chart(frame).encode(
            y=alt.Y("paper:N", title=None, sort=None, axis=alt.Axis(minExtent=210, labelLimit=200)),
            tooltip=[alt.Tooltip("title:N", title="Paper"),
                     alt.Tooltip("citations:Q", title="Citations", format=",")])
        if skewed:
            low, high = max(frame["citations"].min(), 1), frame["citations"].max()
            ticks = [10 ** e for e in range(int(math.log10(low)), int(math.log10(high)) + 2)]
            x = alt.X("citations:Q", title="Citations (logarithmic axis)",
                      scale=alt.Scale(type="symlog", domain=[0 if frame["citations"].min() == 0 else ticks[0],
                                                             ticks[-1]]),
                      axis=alt.Axis(values=ticks, format=",d"))
            # Vega draws text marks black unless told otherwise; use the theme's ink (.streamlit/config.toml).
            ink = "#f0efec" if st.context.theme.type == "dark" else "#0b0b0b"
            chart = base.mark_point(filled=True, size=110, opacity=1).encode(x=x) + base.mark_text(
                align="left", dx=9, color=ink).encode(x=x, text=alt.Text("citations:Q", format=","))
        else:
            chart = base.mark_bar(**BAR).encode(x=alt.X("citations:Q", title="Citations"))
        st.altair_chart(chart.properties(height=28 * len(frame) + 70),
                        alt="Citation counts of the most cited saved papers")
        checked = max(p.citations_checked for p in cited)
        st.caption(("Each step on the axis is ten times the last. " if skewed else "")
                   + f"Semantic Scholar, checked {checked}. Counts were found for {len(cited)} of {len(papers)} "
                   "papers; the newest papers are often not indexed yet. Older papers have had longer to be "
                   "cited, so this is not a quality ranking.")
