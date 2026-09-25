
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "movie_revenue_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #312e81 55%, #7c3aed 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.5rem; }
    .hero p { margin: .5rem 0 0; color: #e5e7eb; }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
    }
    .prediction {
        padding: 1.4rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #ecfdf5, #eff6ff);
        border: 1px solid #bbf7d0;
    }
    .prediction h2 { margin: 0; }
    .small { color: #6b7280; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Discover model categories
# -----------------------------
pre = model.named_steps["preprocessor"]
cat_pipe = pre.named_transformers_["cat"]
encoder = cat_pipe.named_steps["encoder"]
cat_columns = ["Movie_Genre", "Movie_Language", "Movie_Production_Country"]

categories = {
    col: list(vals)
    for col, vals in zip(cat_columns, encoder.categories_)
}

# The training data contains 1,023 distinct genre strings and 401 country
# combinations. For usability, expose the most common-looking / practical
# choices while still allowing an exact custom value.
genre_options = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
    "Family", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western"
]
genre_options = [x for x in genre_options if x in categories["Movie_Genre"]]

language_map = {
    "English (en)": "en", "Hindi (hi)": "hi", "Tamil (ta)": "ta",
    "Telugu (te)": "te", "French (fr)": "fr", "Spanish (es)": "es",
    "German (de)": "de", "Japanese (ja)": "ja", "Korean (ko)": "ko",
    "Chinese (cn)": "cn"
}
language_options = [
    label for label, code in language_map.items()
    if code in categories["Movie_Language"]
]

country_defaults = [
    "[]",
    '[{"iso_3166_1": "US", "name": "United States of America"}]',
    '[{"iso_3166_1": "IN", "name": "India"}]',
    '[{"iso_3166_1": "GB", "name": "United Kingdom"}]',
    '[{"iso_3166_1": "CA", "name": "Canada"}]',
]
country_options = [x for x in country_defaults if x in categories["Movie_Production_Country"]]
if not country_options:
    country_options = categories["Movie_Production_Country"][:5]

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Revenue Predictor</h1>
    <p>Interactive Streamlit interface powered by your trained Random Forest regression pipeline.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🎛️ Movie Inputs")
st.sidebar.caption("Adjust the movie attributes, then click Predict Revenue.")

budget = st.sidebar.number_input(
    "Movie Budget",
    min_value=0.0,
    value=50_000_000.0,
    step=1_000_000.0,
    format="%.0f",
    help="Production budget in the same monetary units used during model training."
)

popularity = st.sidebar.number_input(
    "Movie Popularity",
    min_value=0.0,
    value=20.0,
    step=1.0,
    help="Popularity score from the training dataset."
)

runtime = st.sidebar.slider(
    "Movie Runtime (minutes)",
    min_value=1,
    max_value=300,
    value=120,
    help="Expected movie duration."
)

vote = st.sidebar.number_input(
    "Movie Vote",
    min_value=0.0,
    max_value=10.0,
    value=6.5,
    step=0.1,
    help="Average audience rating."
)

vote_count = st.sidebar.number_input(
    "Movie Vote Count",
    min_value=0,
    value=1000,
    step=100,
    help="Number of audience votes."
)

release_year = st.sidebar.slider(
    "Release Year",
    min_value=1900,
    max_value=2035,
    value=2025,
    help="Movie release year."
)

release_month = st.sidebar.slider(
    "Release Month",
    min_value=1,
    max_value=12,
    value=6,
)

genre_choice = st.sidebar.selectbox(
    "Movie Genre",
    genre_options if genre_options else ["Action"],
)

language_label = st.sidebar.selectbox(
    "Movie Language",
    language_options if language_options else ["English (en)"],
)
language = language_map.get(language_label, language_label)

country_choice = st.sidebar.selectbox(
    "Production Country",
    country_options,
    format_func=lambda x: (
        "No country data"
        if x == "[]"
        else x.replace('[{"iso_3166_1": "', "")
             .split('"')[0]
             if "iso_3166_1" in x else x[:45]
    ),
)

custom_genre = st.sidebar.text_input(
    "Optional: exact trained genre string",
    placeholder="e.g. Action Adventure Comedy",
    help="The model was trained with many combined genre strings. Leave blank to use the selected genre."
)
if custom_genre.strip():
    genre = custom_genre.strip()
else:
    genre = genre_choice

predict = st.sidebar.button("🚀 Predict Revenue", type="primary", use_container_width=True)

# -----------------------------
# Main dashboard
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Details", "🧭 How It Works"])

with tab1:
    st.subheader("Movie Profile")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget", f"${budget:,.0f}")
    c2.metric("Popularity", f"{popularity:,.1f}")
    c3.metric("Runtime", f"{runtime} min")
    c4.metric("Rating", f"{vote:.1f}/10")

    st.divider()

    if predict:
        input_df = pd.DataFrame([{
            "Movie_Budget": budget,
            "Movie_Popularity": popularity,
            "Movie_Runtime": runtime,
            "Movie_Vote": vote,
            "Movie_Vote_Count": vote_count,
            "Release_Year": release_year,
            "Release_Month": release_month,
            "Movie_Genre": genre,
            "Movie_Language": language,
            "Movie_Production_Country": country_choice,
        }])

        try:
            prediction = float(model.predict(input_df)[0])

            st.markdown('<div class="prediction">', unsafe_allow_html=True)
            st.markdown("### 💰 Predicted Movie Revenue")
            st.markdown(f"## ${prediction:,.2f}")
            st.markdown(
                '<p class="small">Prediction generated by the loaded Random Forest regression pipeline.</p>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Input Summary")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the categorical values match the model's training format. "
                "The supplied model uses encoded genre, language and production-country strings."
            )
    else:
        st.info("Set the movie attributes in the sidebar and click **Predict Revenue**.")

        st.subheader("Example Scenario")
        st.write(
            "Try a $100M budget, popularity 30, 130-minute runtime, 7.0 rating, "
            "10,000 votes, a 2025 release, English language and Action genre."
        )

with tab2:
    st.subheader("Trained Model")
    st.write("The uploaded model is a scikit-learn Pipeline containing:")

    model_info = pd.DataFrame({
        "Component": ["Preprocessing", "Numerical preprocessing", "Categorical preprocessing", "Estimator"],
        "Configuration": [
            "ColumnTransformer",
            "Median imputation + StandardScaler",
            "Most-frequent imputation + OneHotEncoder(handle_unknown='ignore')",
            "RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)"
        ]
    })
    st.dataframe(model_info, hide_index=True, use_container_width=True)

    st.subheader("Features Used by the Model")
    feature_info = pd.DataFrame({
        "Feature": [
            "Movie_Budget", "Movie_Popularity", "Movie_Runtime", "Movie_Vote",
            "Movie_Vote_Count", "Release_Year", "Release_Month",
            "Movie_Genre", "Movie_Language", "Movie_Production_Country"
        ],
        "Type": [
            "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
            "Categorical", "Categorical", "Categorical"
        ],
        "Purpose": [
            "Production budget", "Popularity signal", "Runtime in minutes",
            "Average rating", "Number of votes", "Year of release", "Month of release",
            "Genre / genre combination", "Original language", "Production-country combination"
        ]
    })
    st.dataframe(feature_info, hide_index=True, use_container_width=True)

    st.subheader("Training Categories")
    st.write(
        f"Genre categories: **{len(categories['Movie_Genre'])}** · "
        f"Language categories: **{len(categories['Movie_Language'])}** · "
        f"Production-country categories: **{len(categories['Movie_Production_Country'])}**"
    )

with tab3:
    st.subheader("Prediction Workflow")
    st.markdown("""
    **1. User enters movie information**  
    Budget, popularity, runtime, rating, vote count, release date, genre, language and production country.

    **2. Data is assembled into the same feature names used during training**

    **3. The saved preprocessing pipeline transforms the inputs**  
    Numerical values are imputed/scaled and categorical values are one-hot encoded.

    **4. Random Forest Regression predicts revenue**

    **5. The result is displayed interactively**

    **Important:** the prediction is only meaningful within the data distribution and monetary units used to train the original model.
    """)

st.caption("Movie Revenue Prediction • Streamlit • Random Forest Regression")
