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
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
import ollama

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

history = []
recognizer = sr.Recognizer()

# Load the text-to-image pipeline
text_to_image = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
text_to_image.to("cuda")  # Use CUDA

# Initialize the summarization pipeline with explicit model name
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
# Initialize the question-answering pipeline with explicit model name
qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad", device=0)

# Variable to store extracted text from PDF
extracted_text = ""

def clean_text(text):
    # Remove special characters and emojis
    return re.sub(r'[^\w\s]', '', text)

def generate_response(prompt, image_path=None, audio=None, text_to_image_prompt=None):
    if audio:
        with tqdm(total=100, desc="Processing Audio") as pbar:
            with sr.AudioFile(audio) as source:
                audio_data = recognizer.record(source)
                pbar.update(50)
                try:
                    prompt = recognizer.recognize_google(audio_data)
                    pbar.update(50)
                except sr.UnknownValueError:
                    pbar.update(50)
                    return "Sorry, I could not understand the audio.", None, None

    if image_path:
        try:
            with tqdm(total=100, desc="Describing Image") as pbar:
                res = ollama.chat(
                    model="llava",
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Describe this image:',
                            'images': [image_path]
                        }
                    ]
                )
                pbar.update(100)
            response_text = res['message']['content']
        except Exception as e:
            response_text = f"Error describing image: {str(e)}"
    elif text_to_image_prompt:
        try:
            with tqdm(total=50, desc="Generating Image") as pbar:
                images = text_to_image(text_to_image_prompt, num_inference_steps=50).images
                for _ in range(50):
                    pbar.update(1)
            image_path = "generated_image.png"
            images[0].save(image_path)
            response_text = f"Generated an image for the prompt: {text_to_image_prompt}"
        except Exception as e:
            response_text = f"Error generating image: {str(e)}"
    else:
        history.append(prompt)
        final_prompt = "\n".join(history)
        try:
            with tqdm(total=100, desc="Generating Text") as pbar:
                res = ollama.chat(
                    model="gemma2",
                    messages=[
                        {
                            'role': 'user',
                            'content': final_prompt
                        }
                    ]
                )
                pbar.update(100)
            response_text = res['message']['content']
        except Exception as e:
            response_text = f"Error generating text: {str(e)}"
    
    # Clean the response text for voice output
    cleaned_response_text = clean_text(response_text)
    
    with tqdm(total=100, desc="Generating Voice Output") as pbar:
        tts = gTTS(cleaned_response_text)
        tts.save("response.mp3")
        pbar.update(100)

    return response_text, "response.mp3", image_path if text_to_image_prompt else None

# Function to handle document summarization
# Function to handle document summarization
def summarize_document(document):
    global extracted_text
    try:
        doc = PyPDF2.PdfReader(document.name)
        full_text = "".join(page.extract_text() for page in doc.pages if page.extract_text())
        extracted_text = full_text
        chunk_size = 1000
        summaries = []

        for i in tqdm(range(0, len(full_text), chunk_size), desc="Summarizing Document"):
            chunk = full_text[i:i + chunk_size]
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])  # Assuming output is a list of dicts

        return " ".join(summaries)
    except Exception as e:
        return f"Error summarizing document: {str(e)}"

# Function to handle question answering
def answer_question(question):
    try:
        if not extracted_text:
            return "Please upload a document first."
        
        response = qa_model(question=question, context=extracted_text)
        answer = response['answer']

        # Check if the answer is brief or insufficient
        if len(answer.split()) < 20:  # Adjust the threshold as needed
            # Generate explanation using AI model
            explanation_res = ollama.chat(
                model="gemma2",
                messages=[
                    {
                        'role': 'user',
                        'content': f"Why {question}?"
                    }
                ]
            )
            explanation = explanation_res['message']['content']
            return f"Answer: {answer}\nExplanation: {explanation}"
        else:
            return f"Answer: {answer}"
    except Exception as e:
        return f"Error answering question: {str(e)}"

# Function to handle system assistance
def system_assist(text):
    response = "Sorry, I didn't understand that."

    try:
        if "hey jarvis" in text:
            response = "HI SIR AM JARVIS YOUR PERSONAL VOICE ASSISTANT! How can I help you today?"

        elif "what time is it" in text:
            now = datetime.datetime.now()
            response = "The current time is {}:{}:{}".format(now.hour, now.minute, now.second)

        elif "play video " in text:
            search_term = text.replace("play video ", "").strip()
            url = "https://www.youtube.com/results?search_query=" + search_term
            webbrowser.open(url)
            response = "Playing the video on YouTube for {}".format(search_term)

        elif "play music " in text:
            search_term = text.replace("play music ", "").strip()
            url = "https://open.spotify.com/search/" + search_term
            webbrowser.open(url)
            response = "Playing the song on Spotify for {}".format(search_term)

        elif "find" in text:
            search_term = text.replace("find", "").strip()
            url = "https://google.com/search?q=" + search_term
            webbrowser.open(url)
            response = "Here's your search result for {}".format(search_term)

        elif "open notepad" in text:
            subprocess.call(["notepad.exe"])
            response = "Opening Notepad."

        elif "open spotify" in text:
            subprocess.call(["spotify.exe"])
            response = "Opening Spotify."

        elif "open asphalt" in text:
            subprocess.call(["asphalt 9 legends.exe"])
            response = "Opening Asphalt 9 Legends."

        elif "open calculator" in text:
            subprocess.call(["calc.exe"])
            response = "Opening Calculator."

        elif "open paint" in text:
            subprocess.call(["mspaint.exe"])
            response = "Opening Paint."

        elif "shutdown" in text:
            os.system("shutdown /s /t 0")
            response = "Shutting down the PC."

        elif "restart" in text:
            subprocess.call(["shutdown", "/r", "/t", "1"])
            response = "Restarting your device. I will be back once the device gets booted."

        elif "lock" in text:
            os.system("rundll32.exe user32.dll,LockWorkStation")
            response = "Locking your device."

        elif "sleep" in text:
            subprocess.call(["rundll32.exe", "powrprof.dll,SetSuspendState"])
            response = "Putting the PC to sleep."

        elif "location" in text:
            locate = text.replace("location", "").strip()
            response = f"Searching for {locate} on Google Maps"
            webbrowser.open(f"https://www.google.com/maps/place/{locate}")

        elif "qr code" in text:
            qr_text = text.replace("qr code", "").strip()
            img = qrcode.make(qr_text)
            img.save("qrcode.png")
            response = "QR code generated and saved."

        elif "weather " in text:
            city = text.split("in")[-1].strip()
            url = f"https://www.google.com/search?q=weather+{city}"
            webbrowser.open(url)
            response = f"Showing weather information for {city}."

        elif "stop" in text:
            response = "Stopping current processing."

    except Exception as e:
        response = f"Error processing command: {str(e)}"

    return response

# Define Gradio interface for chat
chat_interface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=4, placeholder="Enter your Prompt", label="Prompt"),
        gr.Image(type="filepath", label="Upload Image"),
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Textbox(lines=2, placeholder="Enter Text for Image Generation", label="Text to Image")
    ],
    outputs=["text", "audio", "image"],
    title="Chat Interface",
    description="Interact with the chatbot using text, images, or audio. Enter a prompt to get started."
)

# Define Gradio interface for document summarization
document_interface = gr.Interface(
    fn=summarize_document,
    inputs=gr.File(label="Upload Document"),
    outputs="text",
    title="Document Summarizer",
    description="Upload a document to get a summary of its content."
)

# Define Gradio interface for document question answering
qa_interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=4, placeholder="Ask a question...", label="Question"),
    outputs="text",
    title="Document Q&A",
    description="Ask questions related to the uploaded document. Ensure the document is uploaded first."
)

# Define a new interface for system assistance
system_assist_interface = gr.Interface(
    fn=system_assist,
    inputs=gr.Textbox(lines=4, placeholder="Enter command...", label="System Assist Input"),
    outputs="text",
    title="System Assist",
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

# Combine all interfaces including the new System Assist tab
combined_interface = gr.TabbedInterface(
    [chat_interface, document_interface, qa_interface, system_assist_interface],
    ["Chat Interface", "Document Summarizer", "Document Q&A", "System Assist"],
    theme="default",  # Optional: Set default theme to default mode
    title="Jarvis - AI Assistant"
)

# Launch the interface
combined_interface.launch(share=True)
