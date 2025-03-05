import os
os.environ["HF_HOME"] = "D:/Users/shaha/huggingface"
import torch
import wave
import numpy as np
import sounddevice as sd
import gradio as gr
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel

# Set up Whisper for CPU-based Japanese Speech Recognition
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

# Load Translation Model (Japanese -> English)
# translation_model = "Helsinki-NLP/opus-mt-ja-en"
# translator = pipeline("translation", model=translation_model)

# Load Translation Model (Japanese -> English)
translation_model = "Helsinki-NLP/opus-mt-ja-en"
translator = pipeline("translation", model=translation_model, tokenizer=translation_model)


# Load Local AI Model for Question Answering

# from transformers import AutoModelForCausalLM, AutoTokenizer

# qa_model = "mosaicml/mpt-7b-instruct"
qa_model = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(qa_model, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    qa_model,
    torch_dtype=torch.float32,  # Required for CPU
    trust_remote_code=True  # Necessary for MPT models
)

def process_audio(audio_file):
    """Processes audio file for transcription and translation."""
    japanese_text = transcribe_audio(audio_file)
    english_translation = translate_text(japanese_text)
    return japanese_text, english_translation

def transcribe_audio(filename):
    """Transcribes recorded Japanese speech to text using Whisper."""
    segments, _ = whisper_model.transcribe(filename)
    transcript = " ".join(segment.text for segment in segments)
    return transcript

def translate_text(japanese_text):
    """Translates Japanese text to English using MarianMT."""
    translation = translator(japanese_text)[0]['translation_text']
    return translation

def answer_question(question):
    """Generates AI-based answers using MiniCPM."""
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_question(question):
    """Processes user question and returns AI-generated answer."""
    if question:
        return answer_question(question)
    return "Please enter a question."

# Gradio Interface
audio_interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),  # Removed `source` argument
    outputs=[gr.Textbox(label="Japanese Transcript"), gr.Textbox(label="English Translation")],
    title="Live Meeting Assistant - Speech Translation",
    description="Continuously listen to Japanese speech, transcribe, and translate to English in real-time."
)

qa_interface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(label="Ask AI a Question"),
    outputs=gr.Textbox(label="AI Answer"),
    title="Live Meeting Assistant - Question Answering",
    description="Ask AI a question, and get real-time responses."
)

demo = gr.TabbedInterface([audio_interface, qa_interface], ["Speech Translation", "AI Question Answering"])

demo.launch()


