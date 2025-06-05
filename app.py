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
name, auth_status, username = authenticator.login(form_name="Login", location="main")
st.session_state.auth_status = auth_status

# ----------------- Hide Toolbar & Footer -----------------
st.markdown("""
    <style>
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .st-emotion-cache-z5fcl4 { display: none; }
    .viewerBadge_container__1QSob {display: none !important;}
    .stDeployButton {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Models (cached) -----------------
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

# ----------------- MAIN APP -----------------
if st.session_state.auth_status:
    # Clear placeholder (if any)
    if 'placeholder' in st.session_state:
        st.session_state.placeholder.empty()
        del st.session_state['placeholder']

    st.sidebar.success(f"Logged in as {name}")
    authenticator.logout("Logout", "sidebar")

    st.title("Learnext Course Tagging")
    st.warning("Tag suggestions consume OpenAI API tokens. Please use them efficiently.")

    # ------------- Course Inputs -------------
    st.markdown("#### Single Course Input")
    course_name = st.text_input("Course Name", value=st.session_state.get("course_name", ""))
    description = st.text_area("Description", value=st.session_state.get("description", ""))

    # Save inputs to session state
    st.session_state.course_name = course_name
    st.session_state.description = description

    col1, col2 = st.columns(2)
    tag_clicked = col1.button("Tag Suggestions")
    other_clicked = col2.button("Other Suggestions")

    if tag_clicked or other_clicked:
        if not course_name.strip() or not description.strip():
            st.error("Please fill in both the course name and description.")
        else:
            with st.spinner("Analyzing course content..."):
                if tag_clicked:
                    tags = classifierA.classify(course_name, description)
                else:
                    tags = classifierB.classify(course_name, description)

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
