"""
Smartphone Price Prediction - Streamlit app
Developer : Areeba Tahir (FA24-BED-014)
Supervisor: Dr. Rao Muhammad Adeel Nawab
GitHub    : https://github.com/areebaa063

Loads the trained pipeline (model.pkl) and predicts the price (USD) of a phone
from its specifications. The pipeline contains the imputers, scaler, one-hot
encoder and the best algorithm found by 10-fold cross validation (Extra Trees).
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "model.pkl"
DATA_FILE = "cleaned_all_phones.csv"

# Results of the 10-fold cross validation in the notebook (best algorithm: Extra Trees)
BEST_ALGORITHM = "Extra Trees"
CV_RMSE = 224.83
CV_STD = 33.49

VIDEO_RESOLUTIONS = ["720p", "1080p", "4K", "8K"]
VIDEO_FRAME_RATES = ["30fps", "60fps", "120fps", "240fps", "480fps", "960fps"]
VIDEO_COLUMNS = [f"video_{name}" for name in VIDEO_RESOLUTIONS + VIDEO_FRAME_RATES]

NUMERIC_FEATURES = (
    ["inches", "battery", "ram(GB)", "weight(g)", "storage(GB)",
     "res_width", "res_height", "release_year", "os_version"]
    + VIDEO_COLUMNS
)
CATEGORICAL_FEATURES = ["brand", "os_family", "battery_type"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# --------------------------------------------------------------------------- #
# Same feature preparation as Step 4 of the notebook
# --------------------------------------------------------------------------- #
def os_family_of(os_text):
    os_text = str(os_text).strip()
    for family in ["Android", "iOS", "EMUI", "Tizen"]:
        if os_text.startswith(family):
            return family
    return os_text


def build_features(raw):
    """Convert raw phone columns into the 22 model input columns."""
    features = pd.DataFrame(index=raw.index)

    for column in ["inches", "battery", "ram(GB)", "weight(g)", "storage(GB)"]:
        features[column] = raw[column].astype(float)

    size = raw["resolution"].str.extract(r"(\d+)\s*x\s*(\d+)").astype(float)
    features["res_width"] = size[0]
    features["res_height"] = size[1]

    date = pd.to_datetime(raw["announcement_date"])
    features["release_year"] = date.dt.year + (date.dt.month - 1) / 12

    features["os_family"] = raw["os"].map(os_family_of)
    features["os_version"] = pd.to_numeric(
        raw["os"].str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )

    for column in VIDEO_COLUMNS:
        features[column] = raw[column].astype(int)

    features["brand"] = raw["brand"]
    features["battery_type"] = raw["battery_type"]

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# --------------------------------------------------------------------------- #
# Loading (cached so the files are read only once)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_FILE)


st.set_page_config(page_title="Smartphone Price Predictor", page_icon="📱", layout="centered")

if not (os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE)):
    st.error(
        f"'{MODEL_FILE}' and '{DATA_FILE}' must be in the same folder as app.py. "
        "Run the notebook to create model.pkl, then start the app again."
    )
    st.stop()

model = load_model()
phones = load_reference_data()

brands = sorted(phones["brand"].unique())
resolutions = phones["resolution"].value_counts().index.tolist()      # most common first
storage_options = sorted(phones["storage(GB)"].unique().tolist())
ram_options = sorted(phones["ram(GB)"].unique().tolist())
battery_types = sorted(phones["battery_type"].unique())
os_families = ["Android", "iOS", "EMUI", "Tizen"]

# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("📱 Smartphone Price Predictor")
st.write(
    "Enter the specifications of a phone and the model will estimate its price in US dollars. "
    f"The model is **{BEST_ALGORITHM}**, the best of 5 algorithms compared with 10-fold cross validation."
)

with st.sidebar:
    st.header("About")
    st.write(
        f"**Model:** {BEST_ALGORITHM}\n\n"
        f"**10-fold CV RMSE:** ${CV_RMSE:,.0f} (± ${CV_STD:,.0f})\n\n"
        f"**Trained on:** {len(phones):,} phones, {phones['brand'].nunique()} brands\n\n"
        "**Developer:** Areeba Tahir (FA24-BED-014)\n\n"
        "**Supervisor:** Dr. Rao Muhammad Adeel Nawab"
    )
    st.link_button("GitHub profile", "https://github.com/areebaa063")

with st.form("phone_form"):
    st.subheader("General")
    col1, col2 = st.columns(2)
    brand = col1.selectbox("Brand", brands, index=brands.index("Samsung"), key="brand")
    os_family = col2.selectbox("Operating system", os_families, index=0, key="os_family")

    col1, col2 = st.columns(2)
    os_version = col1.number_input(
        "OS version", min_value=1.0, max_value=20.0, value=12.0, step=1.0,
        help="For example 12 for Android 12 or 15 for iOS 15.",
    )
    year = col2.selectbox("Announcement year", list(range(2016, 2027)), index=6, key="year")
    month_name = col1.selectbox("Announcement month", MONTHS, index=2, key="month")

    st.subheader("Display and body")
    col1, col2, col3 = st.columns(3)
    inches = col1.number_input(
        "Screen size (inches)", min_value=3.0, max_value=12.0, value=6.5, step=0.1
    )
    resolution = col2.selectbox("Resolution (pixels)", resolutions, index=0)
    weight = col3.number_input(
        "Weight (g)", min_value=100.0, max_value=600.0, value=195.0, step=1.0
    )

    st.subheader("Hardware")
    col1, col2, col3, col4 = st.columns(4)
    ram = col1.selectbox("RAM (GB)", ram_options, index=ram_options.index(8))
    storage = col2.selectbox("Storage (GB)", storage_options, index=storage_options.index(128))
    battery = col3.number_input(
        "Battery (mAh)", min_value=1000, max_value=10000, value=5000, step=100
    )
    battery_type = col4.selectbox("Battery type", battery_types, index=battery_types.index("Li-Po"))

    st.subheader("Video recording")
    resolution_defaults = {"720p": True, "1080p": True, "4K": True, "8K": False}
    fps_defaults = {"30fps": True, "60fps": True}
    video_support = {}

    cols = st.columns(len(VIDEO_RESOLUTIONS))
    for col, name in zip(cols, VIDEO_RESOLUTIONS):
        video_support[f"video_{name}"] = col.checkbox(name, value=resolution_defaults[name])

    cols = st.columns(3)
    for index, name in enumerate(VIDEO_FRAME_RATES):
        video_support[f"video_{name}"] = cols[index % 3].checkbox(
            name, value=fps_defaults.get(name, False)
        )

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    announcement_date = f"{year}-{MONTHS.index(month_name) + 1:02d}-01"

    raw_phone = {
        "brand": brand,
        "os": f"{os_family} {os_version:g}",
        "inches": inches,
        "resolution": resolution,
        "battery": battery,
        "battery_type": battery_type,
        "ram(GB)": ram,
        "announcement_date": announcement_date,
        "weight(g)": weight,
        "storage(GB)": storage,
        **video_support,
    }
    feature_vector = build_features(pd.DataFrame([raw_phone]))
    predicted_price = float(model.predict(feature_vector)[0])

    st.divider()
    st.metric("Predicted price", f"${predicted_price:,.2f} USD")
    st.caption(
        f"Typical error of this model (cross-validation RMSE): about ${CV_RMSE:,.0f}. "
        f"So the real price is likely to be roughly between ${max(predicted_price - CV_RMSE, 0):,.0f} "
        f"and ${predicted_price + CV_RMSE:,.0f}."
    )

    if year > 2023:
        st.warning(
            "The model was trained on phones announced between September 2016 and August 2023, "
            "so predictions for later dates are less reliable."
        )
    if os_family == "iOS" and brand != "Apple":
        st.warning("iOS phones in the training data are all made by Apple; this combination is unusual.")

    with st.expander("Show the feature vector sent to the model"):
        st.dataframe(feature_vector.T.astype(str).rename(columns={0: "value"}))
