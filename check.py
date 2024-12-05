import os
import re
import datetime
import subprocess
import webbrowser
import qrcode
import gradio as gr
import PyPDF2
from tqdm import tqdm
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
import torch
import ollama

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Initialize global variables
history = []
recognizer = None
extracted_text = ""

# Load models with GPU optimization and reduced memory usage
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to("cuda")

text_to_image = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16
).to("cuda")

qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad", device=0)

# Helper function to clean text
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

# Main response generation function
def generate_response(prompt, image_path=None, audio=None, text_to_image_prompt=None):
    global recognizer

    # Process audio input
    if audio:
        if recognizer is None:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            try:
                prompt = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return "Sorry, I could not understand the audio.", None, None

    # Handle different response types
    response_text, image_path = handle_response_types(prompt, image_path, text_to_image_prompt)
    
    # Generate audio response
    cleaned_response_text = clean_text(response_text)
    audio_file = save_audio_response(cleaned_response_text)

    return response_text, audio_file, image_path

def handle_response_types(prompt, image_path, text_to_image_prompt):
    if image_path:
        response_text = describe_image(image_path)
    elif text_to_image_prompt:
        response_text, image_path = generate_image(text_to_image_prompt)
    else:
        response_text = chat_with_gemma2(prompt)
    
    return response_text, image_path

def describe_image(image_path):
    try:
        res = ollama.chat(
            model="llava",
            messages=[{'role': 'user', 'content': 'Describe this image:', 'images': [image_path]}]
        )
        return res['message']['content']
    except Exception as e:
        return f"Error describing image: {str(e)}"

def generate_image(text_to_image_prompt):
    try:
        images = text_to_image(text_to_image_prompt, num_inference_steps=25).images
        image_path = "generated_image.png"
        images[0].save(image_path)
        return f"Generated an image for the prompt: {text_to_image_prompt}", image_path
    except Exception as e:
        return f"Error generating image: {str(e)}", None

def chat_with_gemma2(prompt):
    global history
    history.append(prompt)
    final_prompt = "\n".join(history)
    try:
        res = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': final_prompt}])
        return res['message']['content']
    except Exception as e:
        return f"Error generating text: {str(e)}"

def save_audio_response(text):
    audio_file = "response.mp3"
    tts = gTTS(text)
    tts.save(audio_file)
    return audio_file

def summarize_document(document):
    global extracted_text
    try:
        # Use PdfReader directly instead of the incorrect open method
        doc = PyPDF2.PdfReader(document.name)
        full_text = "".join(page.extract_text() for page in doc.pages if page.extract_text())
        extracted_text = full_text
        chunk_size = 1000
        summaries = []

        # Process in chunks for efficiency
        for i in tqdm(range(0, len(full_text), chunk_size), desc="Summarizing Document"):
            chunk = full_text[i:i + chunk_size]
            summary_ids = summarizer_model.generate(summarizer_tokenizer(chunk, return_tensors="pt").input_ids.to("cuda"), max_length=150, min_length=30)
            summaries.append(summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        return " ".join(summaries)
    except Exception as e:
        return f"Error summarizing document: {str(e)}"
def answer_question(question):
    global extracted_text
    if not extracted_text:
        return "Please upload a document first."
    
    response = qa_model(question=question, context=extracted_text)['answer']
    if len(response.split()) < 20:
        explanation_res = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': f"Why {question}?"}])
        explanation = explanation_res['message']['content']
        return f"Answer: {response}\nExplanation: {explanation}"
    
    return f"Answer: {response}"

def system_assist(command):
    command = command.lower()
    response = ""

    # Mapping commands to actions
    command_actions = {
        "hey jarvis": "Hello! How can I assist you today?",
        "what time is it": f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}.",
        "shutdown": "Shutting down the system.",  # Execute shutdown later
        "restart": "Restarting the system.",  # Execute restart later
        "lock": "Locking the workstation.",  # Execute lock later
        "sleep": "Putting the system to sleep.",  # Execute sleep later
        "stop": "Stopping current operation.",
    }

    if "play video" in command:
        search_term = command.replace("play video", "").strip()
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_term}")
        response = f"Searching for videos on '{search_term}' on YouTube."
    elif "play music" in command:
        search_term = command.replace("play music", "").strip()
        webbrowser.open(f"https://open.spotify.com/search/{search_term}")
        response = f"Searching for music on '{search_term}' on Spotify."
    elif "find" in command:
        search_term = command.replace("find", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={search_term}")
        response = f"Searching for '{search_term}' on Google."
    elif "location" in command:
        place = command.replace("location", "").strip()
        webbrowser.open(f"https://www.google.com/maps/search/{place}")
        response = f"Showing location for '{place}' on Google Maps."
    elif "qr code" in command:
        text = command.replace("qr code", "").strip()
        qr_img = qrcode.make(text)
        qr_img.save("qr_code.png")
        response = "QR code generated and saved as 'qr_code.png'."
    elif "weather" in command:
        city = command.replace("weather", "").strip()
        webbrowser.open(f"https://www.google.com/search?q=weather+{city}")
        response = f"Showing weather information for '{city}'."
    elif "open" in command:
        app_map = {
            "notepad": "notepad.exe",
            "spotify": "spotify.exe",
            "asphalt": "asphalt.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
        }
        for app, exe in app_map.items():
            if app in command:
                subprocess.Popen(exe)
                response = f"Opening {app.capitalize()}."
                break
    elif command in command_actions:
        response = command_actions[command]
        # Execute shutdown, restart, lock, sleep logic later if needed
    else:
        response = "Sorry, I didn't understand the command."
    
    # Execute system commands (shutdown, restart, lock, sleep) as needed
    if "shutdown" in command:
        os.system("shutdown /s /t 1")
    elif "restart" in command:
        os.system("shutdown /r /t 1")
    elif "lock" in command:
        os.system("rundll32.exe user32.dll,LockWorkStation")
    elif "sleep" in command:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    
    return response

# Gradio interface setup
chat_interface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(lines=4), gr.Image(type="filepath"), gr.Audio(type="filepath"), gr.Textbox(lines=2)],
    outputs=["text", "audio", "image"],
    title="Chat Interface"
)

document_interface = gr.Interface(
    fn=summarize_document,
    inputs=gr.File(label="Upload Document"),
    outputs="text",
    title="Document Summarizer"
)

qa_interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=4),
    outputs="text",
    title="Document Q&A"
)

system_assist_interface = gr.Interface(
    fn=system_assist,
    inputs=gr.Textbox(lines=4, placeholder="Enter command..."),
    outputs="text",
    title="System Assistant",
    description=(
        "Interact with your system through voice commands. Here are some functionalities you can try:\n\n"
        "- **Hey Jarvis**: Get a greeting from your personal voice assistant.\n"
        "- **What time is it**: Get the current time.\n"
        "- **Play video [search term]**: Opens YouTube and searches for videos related to the search term.\n"
        "- **Play music [search term]**: Opens Spotify and searches for music related to the search term.\n"
        "- **Find [search term]**: Opens Google and searches for the search term.\n"
        "- **Open Notepad**: Opens the Notepad application on Windows.\n"
        "- **Open Spotify**: Opens the Spotify application on Windows.\n"
        "- **Open Asphalt**: Opens Asphalt 9 Legends game on Windows.\n"
        "- **Open Calculator**: Opens the Calculator application on Windows.\n"
        "- **Open Paint**: Opens Microsoft Paint on Windows.\n"
        "- **Shutdown**: Initiates a system shutdown.\n"
        "- **Restart**: Restarts the system.\n"
        "- **Lock**: Locks the workstation.\n"
        "- **Sleep**: Puts the system to sleep.\n"
        "- **Location [place]**: Searches for the specified place on Google Maps.\n"
        "- **QR code [text]**: Generates and saves a QR code with the specified text.\n"
        "- **Weather [city]**: Opens a Google search for the weather in the specified city.\n"
        "- **Stop**: Stops the current processing."
    )
)

gr.TabbedInterface(
    [chat_interface, document_interface, qa_interface, system_assist_interface],
    ["Chat Interface", "Document Summarizer", "Document Q&A", "System Assist"],
    theme="default",  # Optional: Set default theme to default mode
    title=" jarvis -AI Assistant"
).launch(share=True)
