import streamlit as st
from transformers import pipeline

# Load text classification model from Hugging Face
text_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Streamlit app
def highlight_text(text, confidence):
    if confidence > 50:
        return f"<div style='background-color:#FFD700; padding: 5px; border-radius: 5px;'>{text}</div>"
    else:
        return text

def main():
    st.title("AI Content Detection with Text Highlighting")

    # Get user input
    user_input = st.text_area("Enter text for content detection:", "Your text goes here.")

    # Perform content detection when the user clicks the button
    if st.button("Detect Content"):
        with st.spinner("Detecting content..."):
            result = text_classifier(user_input)[0]

        # Display results
        label = "AI" if result['label'] == 'LABEL_1' else "Non-AI"
        confidence = result['score'] * 100
        st.write(f"Detected Label: {label}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Highlight lines more than 50% likely to be AI-generated
        st.write("\nHighlighted Text:")
        lines = user_input.split('\n')
        for line in lines:
            st.markdown(highlight_text(line, confidence), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
