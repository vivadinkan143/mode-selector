import os
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import openai

# ========== CONFIG ==============
DATA_PATH = "data/shipment_data.csv"
LOGO_PATH = "cls_logo.png"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found! Set it as an environment variable called OPENAI_API_KEY.")
    st.stop()

openai.api_key = api_key

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="CLS AI Shipment Mode Selector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== LOAD ASSETS ==========
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Keep only Ground and Air modes
    df = df[df['Carrier Mode'].isin(['Ground', 'Air'])]
    return df

@st.cache_resource
def load_logo():
    return Image.open(LOGO_PATH)

df = load_data()
logo = load_logo()

# ========== HEADER ==========
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=120)
with col2:
    st.markdown(
        """
        <h1 style="color:#004080; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom:0;">
        CLS AI-Powered Shipment Mode Selector
        </h1>
        <p style="color:#555; margin-top:0;">Intelligent logistics optimization powered by AI & Data Science</p>
        """,
        unsafe_allow_html=True
    )
st.markdown("---")

# ========== DATA PREP & MODEL TRAINING ==========
label_encoders = {}
categorical_cols = ["Origin City", "Origin State", "Origin Ctry",
                    "Dest City", "Dest State", "Dest Ctry", "Urgency"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = categorical_cols
target = "Carrier Mode"

X = df[features]
y = df[target]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# ========== USER INPUT FORM ==========
st.header("Enter Shipment Details")

def encode_value(val, col):
    return label_encoders[col].transform([val])[0]

with st.form(key="shipment_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        origin_city = st.selectbox("Origin City", label_encoders["Origin City"].classes_)
        origin_state = st.selectbox("Origin State", label_encoders["Origin State"].classes_)
        origin_ctry = st.selectbox("Origin Country", label_encoders["Origin Ctry"].classes_)

    with col2:
        dest_city = st.selectbox("Destination City", label_encoders["Dest City"].classes_)
        dest_state = st.selectbox("Destination State", label_encoders["Dest State"].classes_)
        dest_ctry = st.selectbox("Destination Country", label_encoders["Dest Ctry"].classes_)

    with col3:
        urgency = st.selectbox("Urgency", label_encoders["Urgency"].classes_)

    submitted = st.form_submit_button("Predict Best Mode")

if submitted:
    try:
        input_features = [
            encode_value(origin_city, "Origin City"),
            encode_value(origin_state, "Origin State"),
            encode_value(origin_ctry, "Origin Ctry"),
            encode_value(dest_city, "Dest City"),
            encode_value(dest_state, "Dest State"),
            encode_value(dest_ctry, "Dest Ctry"),
            encode_value(urgency, "Urgency"),
        ]

        prediction = clf.predict([input_features])[0]
        st.success(f"Recommended Shipment Mode: **{prediction}**")

        # ========== COST COMPARISON ==========
        st.markdown("### Historical Cost Comparison (Ground vs Air)")
        cost_df = df.groupby('Carrier Mode')['Customer Charge USD'].median().reset_index()
        st.dataframe(cost_df.set_index("Carrier Mode"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")

# ========== GPT-4 MINI INTERACTIVE ASSISTANT ==========
st.header("💬 AI Assistant - Ask Anything about Shipment")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def query_gpt4mini(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, we are currently facing a technical issue."

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    answer = query_gpt4mini(user_question)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# ========== FOOTER ==========
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: #777; font-size: 13px;'>
    © 2025 CLS Global Logistics Services | Developed by <strong>Vishnuvardhan Kambikkanam</strong>
    </p>
    """,
    unsafe_allow_html=True,
)
