from transformers import pipeline

def ai_content_detection(text):
    # Load the text classification pipeline
    classifier = pipeline("text-classification", model="bert-base-uncased")

    # Classify the input text
    result = classifier(text)

    # Get the label and score for the top prediction
    label = result[0]['label']
    score = result[0]['score']

    return label, score

if __name__ == "__main__":
    # Example usage
    input_text = "Your AI-generated content goes here."
    
    label, score = ai_content_detection(input_text)

    print(f"AI Content Label: {label}")
    print(f"AI Content Score: {score * 100:.2f}%")
