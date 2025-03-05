# Live-Meeting-Assistant
## Overview
The Live Meeting Assistant is a Python-based application designed to assist in live meetings by providing real-time speech transcription, translation, and question-answering capabilities. The application leverages state-of-the-art machine learning models to transcribe Japanese speech, translate it into English, and answer questions using an AI-based question-answering system.

## Features
1. **Speech Transcription:** Transcribes Japanese speech to text using the Whisper model.

2. **Speech Translation:** Translates transcribed Japanese text into English using the Helsinki-NLP/opus-mt-ja-en model.

3. **Question Answering:** Answers user questions using the google/flan-t5-small model.

4. **Gradio Interface:** Provides an interactive web interface for easy use.

## Installation
### Prerequisites
- Python 3.8 or higher

- pip (Python package installer)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/live-meeting-assistant.git
   cd live-meeting-assistant
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Hugging Face environment:
   Ensure you have the Hugging Face environment variable set up correctly. You can set it in your script as shown below:
   ```
   import os
   os.environ["HF_HOME"] = "D:/Users/shaha/huggingface"
   ```

4. Run the application:
   ```
   python app.py
   ```

## Usage
### Speech Translation
1. **Record or Upload Audio:** Use the Gradio interface to record or upload an audio file containing Japanese speech.

2. **Transcription and Translation:** The application will transcribe the Japanese speech and translate it into English.

3. **View Results:** The transcribed Japanese text and its English translation will be displayed in the interface.

### Question Answering
1. **Enter a Question:** Type your question into the text box provided in the Gradio interface.

2. **Get an Answer:** The AI will process your question and provide an answer in real-time.

## Code Structure
- **app.py:** The main script that sets up the Whisper model, translation pipeline, and question-answering model. It also launches the Gradio interface.

- **requirements.txt:** Lists all the Python dependencies required to run the application.

## Models Used
- **Whisper Model:** Used for speech-to-text transcription.

- **Helsinki-NLP/opus-mt-ja-en:** Used for translating Japanese text to English.

- **google/flan-t5-small:** Used for answering user questions.

## Gradio Interface
The application provides a user-friendly Gradio interface with two main tabs:

1. **Speech Translation:** For transcribing and translating Japanese speech.

2. **AI Question Answering:** For answering user questions using AI.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

## Acknowledgments
[Hugging Face](https://huggingface.co/) for providing the pre-trained models.

[Gradio](https://gradio.app/) for the easy-to-use web interface.

[Whisper](https://github.com/openai/whisper) for the speech-to-text model.

## Contact
For any questions or issues, please open an issue on the GitHub repository or contact the maintainer directly.

> [!NOTE]
> Ensure you have the necessary hardware resources, especially if running on CPU, as some models can be resource-intensive. Also, you can change model for better inference.
