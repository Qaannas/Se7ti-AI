# Apply PyTorch patch to fix compatibility with Streamlit
import torch_patch

import streamlit as st
import pytesseract
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_cpp import Llama
from googletrans import Translator  # pip install googletrans==4.0.0-rc1
from deep_translator import GoogleTranslator
import base64
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv


def moroccan_design():
    st.markdown(
        """
    <style>
    /* Moroccan Zellige-inspired background with light grey */
    .stApp {
        background-color: #31363F;  /* Grey background */         
        background-size: 40px 40px;
    }
    
    /* Moroccan-style containers with light background */
    .stContainer {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Moroccan color palette with red and green */
    h1, h2, h3 {
        color: #D32F2F;  /* Red color */
        text-shadow: 1px 1px 2px rgba(211, 47, 47, 0.3);
    }
    
    .stTextInput > div > div > input {
        background-color: #F1F8E9;  /* Light green background */
        border: 2px solid #388E3C;  /* Green border */
        border-radius: 10px;
        color: #388E3C;  /* Green text */
    }
    
    .stButton > button {
        background-color: #388E3C;  /* Green button */
        color: #FFFFFF;  /* White text */
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #D32F2F;  /* Red on hover */
        transform: scale(1.05);
    }
    
    /* Moroccan-inspired scroll bar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #E5E5E5;  /* Light grey */
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #388E3C;  /* Green thumb */
        border-radius: 10px;
    }
    
    /* Animation for chat messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Cached model loading function with GPU detection
@st.cache_resource
def load_models():
    try:
        # Check if CUDA (GPU) is available
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.info(f"Using device: {device}")
        
        # Initialize the Tiny Llama model
        llm = Llama(
            model_path="/home/fazkes/Desktop/SE7TI-DOCTOR/Se7ti-AI-Darija-Doctor-/tiny-llama-1.1b-chat-medical.q8_0.gguf"
        )

        # Initialize the Darija-to-Arabic model - use GPU if available
        tokenizer_darija_arabic = AutoTokenizer.from_pretrained(
            "tachicart/nllb-ft-darija"
        )
        model_darija_arabic = AutoModelForSeq2SeqLM.from_pretrained(
            "tachicart/nllb-ft-darija"
        ).to(device)

        # Initialize the Terjman-Ultra model for Darija translation - use GPU if available
        tokenizer_darija = AutoTokenizer.from_pretrained("atlasia/Terjman-Ultra")
        model_darija = AutoModelForSeq2SeqLM.from_pretrained("atlasia/Terjman-Ultra").to(device)

        # Initialize the Google Translate API
        translator = Translator()

        return {
            "llm": llm,
            "tokenizer_darija_arabic": tokenizer_darija_arabic,
            "model_darija_arabic": model_darija_arabic,
            "tokenizer_darija": tokenizer_darija,
            "model_darija": model_darija,
            "translator": translator,
            "device": device,
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def clean_text(text):
    """Clean the text by replacing <0x0A> with actual newlines."""
    return text.replace("<0x0A>", "\n")


def translate_to_darija(text):
    try:
        translation = GoogleTranslator(source="en", target="ar").translate(text)
        return translation
    except Exception as e:
        return f"Error during translation: {e}"


def translate_to_darija1(text, models):
    """Translate text to Moroccan Darija using Terjman-Ultra model."""
    try:
        # Get device from models dictionary
        device = models.get("device", "cpu")
        
        # Prepare inputs and move to appropriate device
        inputs = models["tokenizer_darija"](
            text, return_tensors="pt", truncation=True, max_length=512
        )
        
        # Move inputs to the right device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        outputs = models["model_darija"].generate(**inputs, max_length=512)
        
        # Decode the output
        translated_text = models["tokenizer_darija"].decode(
            outputs[0], skip_special_tokens=True
        )
        return clean_text(translated_text)
    except Exception as e:
        st.warning(f"Darija translation error: {e}")
        # Fallback to Google Translate
        try:
            return clean_text(models["translator"].translate(text, dest="ar").text)
        except Exception as fallback_error:
            st.error(f"Fallback translation error: {fallback_error}")
            return text


def translate_darija_to_arabic_to_english(darija_text, models):
    """Translate Darija text to Arabic, then to English."""
    try:
        # Step 1: Translate Darija to Arabic
        inputs = models["tokenizer_darija_arabic"](
            darija_text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = models["model_darija_arabic"].generate(**inputs, max_length=512)
        arabic_translation = models["tokenizer_darija_arabic"].decode(
            outputs[0], skip_special_tokens=True
        )

        # Step 2: Translate Arabic to English using Google Translate API
        english_translation = (
            models["translator"].translate(arabic_translation, src="ar", dest="en").text
        )

        return {
            "Darija Input": clean_text(darija_text),
            "Arabic Translation": clean_text(arabic_translation),
            "English Translation": clean_text(english_translation),
        }
    except Exception as e:
        st.warning(f"Translation error: {e}")
        try:
            # Fallback to direct Google Translate
            english_translation = (
                models["translator"].translate(darija_text, dest="en").text
            )
            return {
                "Darija Input": clean_text(darija_text),
                "Arabic Translation": "Translation failed",
                "English Translation": clean_text(english_translation),
            }
        except Exception as fallback_error:
            st.error(f"Fallback translation error: {fallback_error}")
            return {
                "Darija Input": clean_text(darija_text),
                "Arabic Translation": "Translation failed",
                "English Translation": clean_text(darija_text),
            }


def construct_prompt(messages, model_type="llama"):
    """Construct a prompt for the selected model."""
    if model_type == "llama":
        # Format for Llama model
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"<|user|>\n{message['content']}</s>\n"
            elif message["role"] == "assistant":
                prompt += f"<|assistant|>\n{message['content']}</s>\n"
        prompt += "<|assistant|>\n"
        return prompt
    else:
        # For Gemini, we'll just return the messages in the right format
        return messages


def generate_gemini_response(gemini_model, messages, config=None):
    """Generate a response using the Gemini model."""
    try:
        # Set default config if not provided
        if not config:
            config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        # Generate response with configuration
        generation_config = genai.GenerationConfig(
            temperature=config["gemini_temperature"],
            top_p=config["gemini_top_p"],
            top_k=config["gemini_top_k"],
            max_output_tokens=1024,
        )
        
        response = gemini_model.generate_content(
            gemini_messages,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error generating Gemini response: {e}")
        return "I'm sorry, I couldn't generate a response due to an error."


# Function to extract the name of the medication from an image using OCR
def extract_medicine_name(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    lines = extracted_text.split("\n")
    medicine_name = ""
    for line in lines:
        line = line.strip()
        if line and line.isupper() and not any(char.isdigit() for char in line):
            medicine_name = line
            break
    if not medicine_name:
        for line in lines:
            if line.strip():
                medicine_name = line.strip()
                break
    return medicine_name


# Function to fetch drug details from the OpenFDA API
def fetch_drug_details(medicine_name):
    OPENFDA_API_URL = "https://api.fda.gov/drug/label.json"
    try:
        response = requests.get(
            f"{OPENFDA_API_URL}?search=openfda.brand_name:{medicine_name}&limit=1"
        )
        data = response.json()
        if "results" in data and data["results"]:
            drug_info = data["results"][0]
            name = drug_info.get("openfda", {}).get("brand_name", ["N/A"])[0]
            purpose = drug_info.get("purpose", ["N/A"])[0]
            warnings = drug_info.get("warnings", ["N/A"])[0]
            side_effects = drug_info.get("adverse_reactions", ["N/A"])[0]
            dosage = drug_info.get("dosage_and_administration", ["N/A"])[0]
            return {
                "name": name,
                "purpose": purpose,
                "warnings": warnings,
                "side_effects": side_effects,
                "dosage": dosage,
            }
        else:
            return "No information available for this medication."
    except Exception as e:
        return f"Error fetching data: {e}"


# Function to display and structure the fetched data
def display_medicine_details(details):
    if isinstance(details, dict):
        name = translate_to_darija(f"Medicine Name: {details['name']}")
        purpose = translate_to_darija(f"Purpose: {details['purpose']}")
        warnings = translate_to_darija(f"Warnings: {details['warnings']}")
        side_effects = translate_to_darija(f"Side Effects: {details['side_effects']}")
        dosage = translate_to_darija(f"Dosage: {details['dosage']}")

        st.write(f"### {name}")
        st.write(f"**{purpose}**")
        st.write(f"**{warnings}**")
        st.write(f"**{side_effects}**")
        st.write(f"**{dosage}**")
    else:
        st.write(details)


def chatbot(models, gemini_model=None, model_choice="Tiny Llama (Local)", model_config=None):
    # Apply Moroccan design
    moroccan_design()

    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    # Base64-encoded image (replace with your own encoded string)
    icon_base64 = image_to_base64("Se7ti.ico")

    # Animated HTML header with Moroccan style
    html_code = f"""
    <div style="display: flex; align-items: center; animation: fadeIn 1s ease-out;">
        <img src="data:image/x-icon;base64,{icon_base64}" width="50" style="margin-right: 10px; transform: rotate(0deg); transition: transform 0.5s ease;">  
        <h1 style="margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">Se7ti AI</h1>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    # Display which model is being used
    st.markdown(
        f"""
    <div style="animation: fadeIn 1.5s ease-out; margin-bottom: 20px;">
        <p style="color: #E2DFD0; font-style: italic;">Using model: <strong>{model_choice}</strong></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add a subtle animation to the subtitle
    st.markdown(
        """
    <div style="animation: fadeIn 1.5s ease-out;">
        <p style="color: #E2DFD0; font-style: italic;">tbib bdarija kijawbk ela l2ass2ila ela se7tk</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Add a clear conversation button
    if st.button("Clear Conversation", key="clear_chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Display chat messages from history with animation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input with Moroccan-styled input
    if prompt := st.chat_input("Dkhl So2al diyalk"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Translate the input
        with st.spinner("Tajma3 l3lougha..."):
            translation_result = translate_darija_to_arabic_to_english(prompt, models)
            translated_input = translation_result["English Translation"]

        # Animated translation details
        with st.expander("Tafassil Tajma3 l3lougha"):
            st.write(f"**Darija Input:** {translation_result['Darija Input']}")
            st.write(
                f"**Arabic Translation:** {translation_result['Arabic Translation']}"
            )
            st.write(f"**English Translation:** {translated_input}")

        # Prepare messages for model
        model_messages = [msg for msg in st.session_state.messages[:-1]] + [
            {"role": "user", "content": translated_input}
        ]

        # Generate response with spinner and animation
        with st.chat_message("assistant"):
            try:
                with st.spinner("Se7ti AI kajawbk..."):
                    # Slight delay to simulate thinking
                    time.sleep(1)

                    # Generate response based on selected model
                    if model_choice == "Tiny Llama (Local)":
                        # Construct prompt and generate response using Llama
                        prompt = construct_prompt(model_messages, "llama")
                        output = models["llm"](
                            prompt, 
                            max_tokens=model_config["llama_max_tokens"], 
                            temperature=model_config["llama_temperature"],
                            stop=["</s>"]
                        )
                        response = output["choices"][0]["text"].strip()
                    else:
                        # Generate response using Gemini
                        response = generate_gemini_response(gemini_model, model_messages, model_config)

                    # Translate response to Darija
                    darija_response = translate_to_darija1(response, models)

                    # Animated response
                    st.markdown(darija_response)

                    # Add assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                response = "I'm sorry, I couldn't generate a response."
                st.error(f"Model generation error: {e}")
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


def medication_search():
    # Apply Moroccan design
    moroccan_design()

    st.title("Smiyt dwa 🔎")

    # Animated subtitle
    st.markdown(
        """
    <div style="animation: fadeIn 1.5s ease-out;">
        <p style="color: #E2DFD0; font-style: italic;">7et tswera awla dkhl smiyto bach n3tewk tafassil elih .</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Option to choose input method with Moroccan styling
    input_method = st.radio(
        "Khtar wehda mn hado:",
        ("7et Tswera Diyal Dwa", "Dkhl Smiyt Dwa"),
        help="Ikhtar kifash bghiti tdkhol smiyt edwa",
    )

    medicine_name = None

    if input_method == "7et Tswera Diyal Dwa":
        uploaded_image = st.file_uploader(
            "7et Tswera Diyal Dwa", type=["jpg", "png", "jpeg"]
        )
        if uploaded_image is not None:
            # Animated image display
            st.image(
                uploaded_image,
                caption="Uploaded Image",
                use_column_width=True,
                output_format="PNG",
            )
            with st.spinner("Kaynhll smiyt dwa..."):
                medicine_name = extract_medicine_name(uploaded_image)
    elif input_method == "Dkhl Smiyt Dwa":
        medicine_name = st.text_input("Dkhl Smiyt Dwa:")

    if medicine_name:
        # Animated medicine name display
        st.markdown(
            f"""
        <div style="animation: fadeIn 0.7s ease-out; background-color: rgba(139, 69, 19, 0.1); 
        border-left: 5px solid #8B4513; padding: 10px; margin: 10px 0;">
        **Extracted/Entered Medication Name:** {medicine_name}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Fetch and display drug details with spinner
        with st.spinner("Kayjib l3loughat 3la dwa..."):
            drug_details = fetch_drug_details(medicine_name)
            display_medicine_details(drug_details)
    else:
        st.write("Dkhl hna chi tswera diyal dwa .")


# Function to configure and load Gemini model
def load_gemini_model():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Gemini API key from environment variable
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not gemini_api_key or gemini_api_key.strip() == "":
            # If running for the first time, prompt for API key input
            gemini_api_key = st.text_input(
                "Enter your Google API key to access Gemini:", 
                type="password",
                help="Get your API key from https://makersuite.google.com/app/apikey"
            )
            if not gemini_api_key:
                st.warning("No Google API key provided. Cannot use Gemini model.")
                return None
            else:
                # Save the API key for future runs
                with open(".env", "a") as f:
                    f.write(f"\nGOOGLE_API_KEY={gemini_api_key}")
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Get available models
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Use Gemini Pro model for best results
        model_name = "gemini-pro"  # Using the stable model name
        model = genai.GenerativeModel(model_name)
        
        st.success(f"Successfully loaded Gemini model: {model_name}")
        return model
        
    except Exception as e:
        st.error(f"Error loading Gemini model: {e}")
        return None


# Main function (similar to original, with design updates)
def main():
    # Set up the Streamlit page with Moroccan-inspired icon
    st.set_page_config(page_title="Se7ti BOT", page_icon="Se7ti.ico")
    
    # Sidebar menu with Moroccan styling
    st.sidebar.title("Menu")
    
    # Add model selector in sidebar
    st.sidebar.subheader("Model Settings")
    model_choice = st.sidebar.radio(
        "Choose AI Model:",
        ["Tiny Llama (Local)", "Google Gemini (API)"],
        help="Select which AI model to use for generating responses"
    )
    
    # Model configuration options
    with st.sidebar.expander("Model Configuration"):
        if model_choice == "Tiny Llama (Local)":
            llama_max_tokens = st.slider("Max Output Tokens", 100, 1000, 500, 50)
            llama_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        else:
            gemini_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            gemini_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
            gemini_top_k = st.slider("Top K", 1, 40, 40, 1)
    
    # Load appropriate models based on selection
    models = None
    gemini_model = None
    model_config = {
        "llama_max_tokens": 500,
        "llama_temperature": 0.7,
        "gemini_temperature": 0.7,
        "gemini_top_p": 0.9,
        "gemini_top_k": 40
    }
    
    # Update model config with slider values
    if model_choice == "Tiny Llama (Local)":
        model_config["llama_max_tokens"] = llama_max_tokens
        model_config["llama_temperature"] = llama_temperature
    else:
        model_config["gemini_temperature"] = gemini_temperature
        model_config["gemini_top_p"] = gemini_top_p
        model_config["gemini_top_k"] = gemini_top_k
    
    if model_choice == "Tiny Llama (Local)":
        # Load Llama models
        models = load_models()
        if not models:
            st.error("kayn chi mouchkil flmodel, 3awd t2eked bila kolchi m9Ad")
            return
    else:  # Gemini model
        # Load translation models first
        models = load_models()
        if not models:
            st.error("kayn chi mouchkil flmodel, 3awd t2eked bila kolchi m9Ad")
            return
            
        # Then load Gemini model
        gemini_model = load_gemini_model()
        if not gemini_model:
            st.error("kayn chi mouchkil flmodel dyial Gemini, 3awd t2eked bil API key s7i7")
            return
    
    # Main menu options
    options = st.sidebar.radio(
        "Ikhtar wahd men hadchi:",
        ["Ma3loumat 3la Dawa", "Se7ti BOT"],
        help="Ikhtar l-khedma lli bghiti",
    )

    if options == "Ma3loumat 3la Dawa":
        medication_search()
    elif options == "Se7ti BOT":
        chatbot(models, gemini_model, model_choice, model_config)


if __name__ == "__main__":
    main()
