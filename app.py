import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib


pipe_lr = joblib.load("text_emotion.pkl")


emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "😊", "joy": "😂",
    "neutral": "😐", "sad": "😔", "shame": "😳", "surprise": "😮"
}


def predict_emotions(text):
    return pipe_lr.predict([text])[0]


def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])


st.set_page_config(page_title="Emotion Detector", layout="wide")


st.title("💬 Emotion Detection App")
st.markdown("#### Analyze and understand emotions from text instantly!")


with st.sidebar:
    st.subheader("📝 How It Works:")
    st.info(
        """
        1. Enter any text in the box.
        2. Click on 'Analyze Emotion' to predict emotions.
        3. Get the predicted emotion with confidence score.
        4. Explore probability distribution of other emotions.
        """
    )
    st.caption("💡 Built with Scikit-Learn & Streamlit")


st.subheader("📚 Enter Your Text Below")
raw_text = st.text_area("Type your text here...", height=150)


if st.button("🔍 Analyze Emotion"):
    if raw_text.strip():
        
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        
        col1, col2 = st.columns(2)

        with col1:
            st.success("🎯 **Predicted Emotion:**")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{prediction} {emoji_icon}</h2>", unsafe_allow_html=True)
            
            st.info(f"📊 Confidence Level: **{np.max(probability) * 100:.2f}%**")

        with col2:
            st.success("📈 **Probability Distribution:**")

            # Create Pie Chart Data
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            # Plot Pie Chart
            pie_chart = alt.Chart(proba_df_clean).mark_arc(innerRadius=50).encode(
                theta="Probability",
                color="Emotion",
                tooltip=["Emotion", "Probability"]
            ).properties(width=300, height=300)

            st.altair_chart(pie_chart, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text for analysis!")

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit | Dev by [Devendra Khachane](#)")
