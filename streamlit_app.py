"""Entry point: `streamlit run streamlit_app.py`."""

import streamlit as st

st.set_page_config(page_title="Research assistant", page_icon=":material/school:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/discover.py", title="Discover", icon=":material/travel_explore:", default=True,
                help="Search arXiv, follow topics on a watchlist, and get new papers ranked against your "
                     "research interests. Save the ones worth reading."),
        st.Page("app_pages/library.py", title="Library", icon=":material/library_books:",
                help="Every paper you saved, with reading status and tags. Filter, open a paper in the "
                     "reader, or export BibTeX."),
        st.Page("app_pages/reader.py", title="Reader", icon=":material/menu_book:",
                help="Read one paper: a structured digest with quotes checked against the text, questions "
                     "answered from cited passages, the full text and your notes."),
        st.Page("app_pages/synthesize.py", title="Synthesize", icon=":material/compare_arrows:",
                help="Compare two to five saved papers side by side and draft a related-work paragraph "
                     "that cites them."),
        st.Page("app_pages/insights.py", title="Insights", icon=":material/insights:",
                help="Charts computed from your library: submissions over time, categories, terms, "
                     "topic map, co-author network and citation counts."),
    ],
    position="top",
)

st.title(page.title, icon=page.icon)
page.run()
