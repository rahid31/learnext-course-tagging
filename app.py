import streamlit as st
# -------------------- Page Setup --------------------
page_icon = "data/image/lx_icon_192.png"
company_logo = "data/image/lx_primary_256.png"
st.set_page_config(page_title="Learnext Tagging", page_icon=page_icon, layout="centered")

import pandas as pd
import io
import streamlit_authenticator as stauth
import pickle
from course_tagging import TagClassifier, OtherClassifier, BulkTagging


# -------------------- Global CSS --------------------
st.markdown("""
    <style>
    button[title="Copy to clipboard"],
    button[title="View fullscreen"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Authentication --------------------
credentials = {
    "usernames": {}
}

for username, info in st.secrets["credentials"].items():
    credentials["usernames"][username] = {
        "name": info["name"],
        "password": info["password"]
    }

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="learnext_cookie",
    key="learnext_signature",
    cookie_expiry_days=7
)

if "auth_status" not in st.session_state:
    st.session_state.auth_status = None

# Placeholder for showing logo
placeholder = st.empty()

if st.session_state.auth_status is None:
    with placeholder.container():
        st.image(company_logo, width=192)

# Show login widget outside of placeholder container
name, auth_status, username = authenticator.login("Login", location="main")
st.session_state.auth_status = auth_status

# -------------------- Main App --------------------
if st.session_state.auth_status:
    placeholder.empty()

    #Hide Toolbar & footer
    st.markdown("""
        <style>
        header { visibility: hidden; }
        footer { visibility: hidden; }
        .st-emotion-cache-z5fcl4 { display: none; }  /* Hides Streamlit toolbar */
        .viewerBadge_container__1QSob {display: none !important;} /* Hides 'Created by' and 'Hosted by' */
        .stDeployButton {display: none !important;} /* Hides deploy/share button */
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.success(f"Logged in as {name}")
    authenticator.logout("Logout", "sidebar")

    st.title("Learnext Course Tagging")
    st.warning("Tag suggestions consume OpenAI API tokens. Please use them efficiently.")

    # -------------------- Load Models --------------------
    @st.cache_resource
    def get_classifier_a():
        return TagClassifier()

    @st.cache_resource
    def get_classifier_b():
        return OtherClassifier()
    
    @st.cache_resource
    def get_classifier_c():
        return BulkTagging()

    classifierA = get_classifier_a()
    classifierB = get_classifier_b()
    classifierC = get_classifier_c()

    # -------------------- Course Inputs --------------------
    st.markdown("#### Single Course Input")
    st.session_state.course_name = st.text_input("Course Name", value=st.session_state.get("course_name", ""))
    st.session_state.description = st.text_area("Description", value=st.session_state.get("description", ""))

    col1, col2 = st.columns(2)
    tag_clicked = col1.button("Tag Suggestions")
    other_clicked = col2.button("Other Suggestions")

    if tag_clicked:
        if not st.session_state.course_name.strip() or not st.session_state.description.strip():
            st.error("Please fill in both the course name and description.")
        else:
            with st.spinner("Analyzing course content..."):
                tags = classifierA.classify(st.session_state.course_name, st.session_state.description)
            st.subheader("Suggested Tags")
            if tags:
                st.markdown("\n".join(f"- **{tag}**" for tag in tags))
            else:
                st.info("No relevant tags were identified.")

    if other_clicked:
        if not st.session_state.course_name.strip() or not st.session_state.description.strip():
            st.error("Please fill in both the course name and description.")
        else:
            with st.spinner("Analyzing course content..."):
                tags = classifierB.classify(st.session_state.course_name, st.session_state.description)
            st.subheader("Suggested Tags")
            if tags:
                st.markdown("\n".join(f"- **{tag}**" for tag in tags))
            else:
                st.info("No relevant tags were identified.")

    # -------------------- Bulk Upload Section --------------------
    st.markdown("---")
    st.markdown("#### Bulk Upload (Optional)")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

            if st.button("Run Bulk Tagging"):
                with st.spinner("Processing bulk tagging..."):
                    result_df = classifierC.classify_bulk(df, name_col="name", desc_col="description")

                st.success("Tagging completed.")
                st.dataframe(result_df.head())

                # Prepare file for download
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Tagged Results",
                    data=csv_buffer.getvalue(),
                    file_name="tagged_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

    sample_data = pd.read_csv("data/sample_data.csv")
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_data.csv",
        mime="text/csv"
    )

elif st.session_state.auth_status is False:
    st.error("Username or password is incorrect")
