import gradio as gr
import joblib

# Load model and vectorizer
model = joblib.load("job_scam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_job(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    if prediction == 1:
        return "⚠ This Job Post Looks Like a SCAM"
    else:
        return "✅ This Job Post Looks Legitimate"

interface = gr.Interface(
    fn=predict_job,
    inputs="text",
    outputs="text",
    title="AI Job Scam Detector",
    description="Enter job description to check if it is Fake or Real."
)

interface.launch()
input("Press Enter to exit...")