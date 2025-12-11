# app.py
import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# Keras / TF imports (keep them together so import errors are obvious)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    # Let the user see an explanatory message inside Streamlit if TF import fails
    st.set_page_config(page_title="Multilingual Language Detector", layout="centered")
    st.title("🌍 Multilingual Language Detector")
    st.error("TensorFlow / Keras import failed. Check your Python environment.")
    st.exception(e)
    st.stop()

# -----------------------------------
# Streamlit Page Config (FIRST LINE)
# -----------------------------------
st.set_page_config(page_title="Multilingual Language Detector", layout="centered")

APP_DIR = Path(__file__).parent.resolve()  # folder where app.py lives
MODEL_PATH = APP_DIR / "language_model.h5"
TOKENIZER_PATH = APP_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = APP_DIR / "label_encoder.pkl"

# -----------------------------------
# Load Model, Tokenizer, LabelEncoder
# -----------------------------------
@st.cache_resource
def load_all():
    # Check files exist and raise informative error if not
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not TOKENIZER_PATH.exists():
        missing.append(str(TOKENIZER_PATH))
    if not LABEL_ENCODER_PATH.exists():
        missing.append(str(LABEL_ENCODER_PATH))
    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n" + "\n".join(missing)
        )

    # Load model and pickles
    model = load_model(str(MODEL_PATH))
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # Sanity checks
    if not hasattr(tokenizer, "texts_to_sequences"):
        raise ValueError("Loaded tokenizer doesn't look like a Keras tokenizer.")
    if not hasattr(label_encoder, "inverse_transform"):
        raise ValueError("Loaded label_encoder doesn't have inverse_transform().")

    return model, tokenizer, label_encoder


# Try load and show friendly error in the UI if something fails
try:
    model, tokenizer, label_encoder = load_all()
except Exception as err:
    st.title("🌍 Multilingual Language Detector")
    st.error("Failed to load required model/tokenizer/encoder files.")
    st.info("Make sure these files are in the same folder as app.py:")
    st.write("- language_model.h5\n- tokenizer.pkl\n- label_encoder.pkl")
    st.exception(err)
    st.stop()

# model specifics
maxlen = 40  # keep this consistent with training

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("🌍 Multilingual Language Detector")
st.write("Enter any text and the model will detect the language. (Top-3 shown)")

# Input box (multiline)
user_input = st.text_area("Enter some text:", height=120)

def predict_language(text, top_k=3):
    # Preprocess
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    pred = model.predict(seq, verbose=0)  # shape (1, n_classes)
    if pred.ndim == 2:
        probs = pred[0]
    else:
        probs = np.array(pred).reshape(-1)

    # Top-k indices
    top_idx = np.argsort(probs)[::-1][:top_k]
    labels = label_encoder.inverse_transform(top_idx)
    top_probs = probs[top_idx]
    return list(zip(labels, top_probs))

# Predict button
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        try:
            results = predict_language(user_input, top_k=3)
            st.success(f"Top prediction: **{results[0][0]}** — {results[0][1]*100:.2f}%")
            st.markdown("**Top-3 predictions:**")
            for label, p in results:
                st.write(f"- {label}: {p*100:.2f}%")
        except Exception as e:
            st.error("Prediction failed — check tokenizer/model compatibility and maxlen.")
            st.exception(e)
