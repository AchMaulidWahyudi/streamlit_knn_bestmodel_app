
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Fixed best model configuration
# -------------------------------
BEST_MODEL_CONFIG = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "manhattan"
}

# Simulated list of 26 features (replace with actual top 26 feature names)
selected_features = ['Feature_' + str(i) for i in range(1, 27)]

# Dummy model placeholder (replace with your trained model if available)
def load_best_model():
    model = KNeighborsClassifier(
        n_neighbors=BEST_MODEL_CONFIG["n_neighbors"],
        weights=BEST_MODEL_CONFIG["weights"],
        metric=BEST_MODEL_CONFIG["metric"]
    )
    # Model should be trained outside and loaded here (joblib.load)
    return model

model = load_best_model()

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(page_title="Deteksi Phishing Website", layout="wide")
st.title("Aplikasi Deteksi Phishing Website")
st.markdown("Model: **KNN (26 fitur, k=5, weights=distance, metric=manhattan)**")

menu = st.sidebar.radio("Navigasi", ["Prediksi Manual / Upload", "Uji Data Testing Baru"])

# -------------------------------
# Halaman 1: Input Manual / Upload CSV
# -------------------------------
if menu == "Prediksi Manual / Upload":
    st.subheader("Input Manual Fitur Website")

    user_input = {}
    col1, col2 = st.columns(2)
    for idx, feature in enumerate(selected_features):
        with (col1 if idx % 2 == 0 else col2):
            user_input[feature] = st.selectbox(f"{feature}", [-1, 0, 1])

    if st.button("Prediksi"):
        df_input = pd.DataFrame([user_input])
        prediction = model.predict(df_input)[0]
        result = "Phishing Website" if prediction == -1 else "Legitimate Website"
        st.success(f"Hasil Prediksi: **{result}**")

    st.divider()
    st.subheader("Upload File CSV untuk Prediksi Massal")
    uploaded_file = st.file_uploader("Unggah file CSV dengan 26 fitur", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing_features = [feat for feat in selected_features if feat not in df.columns]
            if missing_features:
                st.error(f"Kolom berikut tidak ditemukan dalam file: {missing_features}")
            else:
                predictions = model.predict(df[selected_features])
                df['Prediksi'] = ["Phishing" if p == -1 else "Legitimate" for p in predictions]
                st.dataframe(df)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# -------------------------------
# Halaman 2: Uji Coba Data Testing
# -------------------------------
elif menu == "Uji Data Testing Baru":
    st.subheader("Evaluasi Model terhadap Data Testing Baru")

    test_file = st.file_uploader("Unggah data testing (dengan kolom 'Result')", type=["csv"], key="testdata")
    if test_file:
        try:
            test_df = pd.read_csv(test_file)
            if 'Result' not in test_df.columns:
                st.error("Kolom 'Result' tidak ditemukan dalam file.")
            else:
                X_test = test_df[selected_features]
                y_true = test_df['Result']
                y_pred = model.predict(X_test)

                st.write("### Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=["Legitimate", "Phishing"],
                            yticklabels=["Legitimate", "Phishing"])
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Kelas Sebenarnya")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
