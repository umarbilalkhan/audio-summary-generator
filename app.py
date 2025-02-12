import streamlit as st
import whisper
import os
import tempfile
from transformers import pipeline

# Load summarization model (Local & Free)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to transcribe audio using Whisper (Local & Free)
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Runs locally
    result = model.transcribe(audio_file)
    return result["text"]

# Function to summarize text with bullet points
def generate_summary(transcript):
    # Truncate transcript to prevent exceeding model token limits
    truncated_text = " ".join(transcript.split()[:1024])  

    summary = summarizer(truncated_text, max_length=300, min_length=50, do_sample=False)
    
    # Ensure bullet formatting
    bullet_summary = summary[0]["summary_text"].split(". ")
    formatted_summary = "\n- " + "\n- ".join(bullet_summary)  # Convert sentences to bullet points

    return formatted_summary

# Function to save summary to a text file
def save_summary(summary):
    file_path = "summary.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return file_path

# Main Streamlit app
def main():
    st.title("Free Local Audio-to-Bullet Summary Web App")
    st.write("Record audio, transcribe it, and generate a bullet-point summary.")

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name
        
        st.write("### Transcribing...")
        transcript = transcribe_audio(temp_audio_path)
        st.text_area("Transcription:", transcript, height=200)
        
        st.write("### Generating Bullet-Point Summary...")
        summary = generate_summary(transcript)
        st.text_area("Summary:", summary, height=200)
        
        summary_file = save_summary(summary)
        st.download_button("Download Summary", data=open(summary_file, "r").read(), file_name="summary.txt")

if __name__ == "__main__":
    main()
