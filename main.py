import streamlit as st
import pytesseract
from PIL import Image
import requests
import base64
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Fake patient data template
fake_patient_data = {
    "name": "Anas",
    "age": 45,
    "gender": "Male",
    "symptoms": ["persistent cough", "fever", "fatigue"],
    "medical_history": ["hypertension", "allergy to penicillin"],
    "current_medications": ["lisinopril", "paracetamol"],
}

def moroccan_design():
    st.markdown(
        """
    <style>
    .stApp {
        background-color: #31363F;
    }
    
    h1, h2, h3 {
        color: #D32F2F;
        text-shadow: 1px 1px 2px rgba(211, 47, 47, 0.3);
    }
    
    .stButton > button {
        background-color: #388E3C;
        color: #FFFFFF;
        border-radius: 10px;
    }
    
    .stButton > button:hover {
        background-color: #D32F2F;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Cached model loading function
@st.cache_resource
def load_models():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Google Gemini API key from environment variable or prompt user
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key or gemini_api_key.strip() == "":
            # If running for the first time, prompt for API key input
            gemini_api_key = st.text_input(
                "Enter your Google Gemini API key:", 
                type="password",
                help="Get your API key from https://ai.google.dev/"
            )
            if not gemini_api_key:
                st.warning("No Gemini API key provided. The application will not work properly.")
                return None
            else:
                # Save the API key for future runs
                with open(".env", "w") as f:
                    f.write(f"GOOGLE_API_KEY={gemini_api_key}")
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        return {
            "model": model
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def clean_text(text):
    """Clean the text by replacing <0x0A> with actual newlines."""
    return text.replace("<0x0A>", "\n")

# Function to modify patient data
def modify_patient_data():
    global fake_patient_data
    st.sidebar.title("Patient Data Management")

    # Display current data
    st.sidebar.write("### Current Patient Data")
    for key, value in fake_patient_data.items():
        st.sidebar.write(f"{key.capitalize()}: {value}")

    option = st.sidebar.selectbox(
        "Choose an action", ["Modify Data", "Delete Data", "Keep Current Data"]
    )

    if option == "Modify Data":
        key = st.sidebar.selectbox("Which field to modify?", fake_patient_data.keys())
        if key in ["symptoms", "medical_history", "current_medications"]:
            new_value = st.sidebar.text_area(
                f"New {key.capitalize()} (comma-separated)",
                value=", ".join(fake_patient_data[key]),
            )
            fake_patient_data[key] = [item.strip() for item in new_value.split(",") if item.strip()]
        else:
            new_value = st.sidebar.text_input(
                f"New {key.capitalize()}", value=fake_patient_data[key]
            )
            fake_patient_data[key] = new_value
        st.sidebar.write(f"{key.capitalize()} updated!")

    elif option == "Delete Data":
        if st.sidebar.button("Delete All Patient Data"):
            fake_patient_data = {
                "name": "",
                "age": "",
                "gender": "",
                "symptoms": [],
                "medical_history": [],
                "current_medications": [],
            }
            st.sidebar.write("Patient data deleted!")

    return fake_patient_data

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
        st.write(f"### Medicine Name: {details['name']}")
        st.write(f"**Purpose:** {details['purpose']}")
        st.write(f"**Warnings:** {details['warnings']}")
        st.write(f"**Side Effects:** {details['side_effects']}")
        st.write(f"**Dosage:** {details['dosage']}")
    else:
        st.write(details)

def generate_medical_response(query, model, patient_data):
    """Generate a medical response using Gemini AI with patient context.
    
    Uses the model itself to classify if a query is medical or casual, then
    responds appropriately with the right level of detail and context.
    """
    # First do a quick, basic check with the simplified function
    # This is just a fast pre-filter for obvious cases
    is_obvious_casual = not is_medical_query(query)
    
    # This unified prompt lets the model classify and respond to the query in one step
    prompt = f"""You are Se7ti, a highly specialized AI medical assistant developed to provide helpful, accurate, and ethical medical advice. Your core responsibility is to assist patients by providing reliable medical information.

PATIENT PROFILE:
- Name: {patient_data['name']}
- Age: {patient_data['age']} years
- Gender: {patient_data['gender']}
- Current Symptoms: {', '.join(patient_data['symptoms'])}
- Medical History: {', '.join(patient_data['medical_history'])}
- Current Medications: {', '.join(patient_data['current_medications'])}

Step 1: Determine if the user query is medical or health-related in nature:
- Medical queries include: symptoms, diseases, treatments, medications, medical procedures, health concerns, wellness advice, etc.
- Non-medical queries include: general chit-chat, greetings, personal questions about you, questions about other topics like technology, entertainment, etc.

Step 2: Respond appropriately based on your classification:

If MEDICAL:
1. Provide accurate medical information and advice based on current medical consensus
2. Always clarify that your advice does not replace professional medical consultation
3. Show empathy and understanding in your responses
4. When recommending medications, note potential side effects, contraindications with their current medications, and when to seek medical attention
5. Do not diagnose but explain what symptoms might indicate and when to see a doctor
6. Reference the patient's history and current medications when relevant
7. Format your response clearly with sections and bullet points where appropriate

If NON-MEDICAL:
1. Provide a brief, friendly response to the query
2. Gently remind the user that you're specialized in medical topics
3. If appropriate, ask if they have any health-related questions you can help with
4. Keep your response concise and don't attempt to be an expert on non-medical topics

User Query: {query}

Your response (in English):"""
    
    try:
        # Generate response using the unified prompt
        response = model.generate_content(prompt)
        
        # For tracking purposes in the UI (debug mode)
        # We use the basic check's result as an estimate, though the model does the real classification
        is_medical = not is_obvious_casual
        
        # Add debug info
        debug_info = f"Debug: Query '{query}' initially classified as {'CASUAL' if is_obvious_casual else 'MEDICAL'} (final classification by LLM)"
        st.session_state.debug_info = debug_info
        
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response due to a technical error."

def is_medical_query(query):
    """Determine if the query is medical in nature or just casual conversation.
    
    Uses a simple check for common patterns instead of relying on keyword lists.
    The main classification happens in the generate_medical_response function
    where the LLM handles the full classification with context.
    """
    # Handle empty queries
    if not query.strip():
        return False
        
    # Quick check for very obvious non-medical queries
    query = query.lower().strip()
    
    # Very obvious greetings and casual conversation starters
    obvious_casual_patterns = [
        'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 
        'who are you', 'what can you do', 'how does this work'
    ]
    
    # Check for exact matches or if query starts with an obvious casual phrase
    query_words = query.split()
    if query in obvious_casual_patterns or (len(query_words) > 0 and query_words[0] in obvious_casual_patterns):
        return False
        
    # For ambiguous queries, let the LLM decide in the generate_medical_response function
    # The detailed prompting there will handle the nuanced classification
    
    # For longer queries that aren't obvious greetings, default to medical so they get
    # processed by the medical prompt, which includes handling for non-medical topics
    return True

def chatbot(models):
    # Apply Moroccan design
    moroccan_design()

    # Update patient data
    patient_data = modify_patient_data()

    # Add debug mode toggle in the sidebar
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False, help="Show how queries are classified")
    
    if debug_mode:
        st.sidebar.info("Debug mode enabled. You'll see how your messages are classified.")
    
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    # Base64-encoded image
    icon_base64 = image_to_base64("Se7ti.ico")

    # Header with icon
    html_code = f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/x-icon;base64,{icon_base64}" width="50" style="margin-right: 10px;">  
        <h1 style="margin: 0;">Health AI</h1>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    st.markdown("<p style='color: #E2DFD0; font-style: italic;'>Your personal medical assistant</p>", unsafe_allow_html=True)

    # Display patient context reminder
    if any(patient_data['symptoms']):
        st.info(f"Currently providing advice for: {patient_data['name']}, {patient_data['age']} years old, with symptoms: {', '.join(patient_data['symptoms'])}")
    else:
        st.info("I can answer general questions about health or provide personalized medical advice based on patient information in the sidebar.")
    
    # Add scope note
    st.markdown("""
    > **Note**: This assistant is designed specifically for health-related questions. For other topics, please use a general-purpose assistant.
    """)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Enter your question or greeting"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating response..."):
                    # Generate response using Gemini with patient data
                    response = generate_medical_response(prompt, models["model"], patient_data)

                    # Show response
                    st.markdown(response)

                    # Add assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    
                    # Show debug info if debug mode is enabled
                    if debug_mode and "debug_info" in st.session_state:
                        st.info(st.session_state.debug_info)

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

    st.title("Medication Information 🔎")
    st.markdown("<p style='color: #E2DFD0; font-style: italic;'>Upload an image or enter the name of a medication to get details</p>", unsafe_allow_html=True)

    # Option to choose input method
    input_method = st.radio(
        "Choose an option:",
        ("Upload Medication Image", "Enter Medication Name"),
        help="Choose how you want to search for medication information"
    )

    medicine_name = None

    if input_method == "Upload Medication Image":
        uploaded_image = st.file_uploader(
            "Upload medication image", type=["jpg", "png", "jpeg"]
        )
        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption="Uploaded Image",
                use_column_width=True,
                output_format="PNG",
            )
            with st.spinner("Extracting medication name..."):
                medicine_name = extract_medicine_name(uploaded_image)
    elif input_method == "Enter Medication Name":
        medicine_name = st.text_input("Enter medication name:")

    if medicine_name:
        st.markdown(f"**Extracted/Entered Medication Name:** {medicine_name}")

        # Fetch and display drug details
        with st.spinner("Fetching medication information..."):
            drug_details = fetch_drug_details(medicine_name)
            display_medicine_details(drug_details)
    else:
        st.write("Please upload an image or enter a medication name.")

# Main function
def main():
    # Set up the Streamlit page
    st.set_page_config(page_title="Health AI", page_icon="Se7ti.ico")

    # Load models
    models = load_models()
    if not models:
        st.error("There was a problem loading the model. Please check your API key.")
        
        # Show helpful instructions
        st.markdown("""
        ## How to fix this issue:
        
        1. You need a Google Gemini API key
        2. Go to [Google AI Studio](https://ai.google.dev/) and sign up
        3. Get your API key from the console
        4. Restart the application and enter your API key when prompted
        
        Alternatively, make sure your `.env` file has a valid token:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```
        """)
        return

    # Sidebar menu
    st.sidebar.title("Menu")
    options = st.sidebar.radio(
        "Choose an option:",
        ["Medication Information", "Medical Assistant"],
        help="Select which service you want to use"
    )

    if options == "Medication Information":
        medication_search()
    elif options == "Medical Assistant":
        chatbot(models)

if __name__ == "__main__":
    main()
