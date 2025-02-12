import streamlit as st
import whisper
import os
import tempfile
from transformers import pipeline

# Use a lightweight Whisper model
def transcribe_audio(audio_file):
    model = whisper.load_model("tiny")  # Whisper Tiny is lightweight
    result = model.transcribe(audio_file)
    return result["text"]

# Use a smaller summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Generate bullet-point summary
def generate_summary(transcript):
    summary = summarizer(transcript, max_length=300, min_length=50, do_sample=False)
    
    # Ensure bullet formatting
    bullet_summary = summary[0]["summary_text"].replace(". ", ".\n- ")
    return f"- {bullet_summary}"

# Save summary to a text file
def save_summary(summary):
    file_path = "summary.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return file_path

# Streamlit app UI
def main():
    st.title("Audio-to-Bullet Summary (Free & Local)")
    st.write("Upload an audio file, transcribe it, and generate a bullet-point summary.")

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name
        
        st.write("### Transcribing...")
        transcript = transcribe_audio(temp_audio_path)
        st.text_area("Transcription:", transcript, height=200)
        
        st.write("### Generating Summary...")
        summary = generate_summary(transcript)
        st.text_area("Summary:", summary, height=200)
        
        summary_file = save_summary(summary)
        st.download_button("Download Summary", data=open(summary_file, "r").read(), file_name="summary.txt")

if __name__ == "__main__":
    main()
