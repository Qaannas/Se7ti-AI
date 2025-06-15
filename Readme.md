# Se7ti-AI: Medical Assistance and Medication Information App

## Overview

Se7ti-AI is a medical assistance application designed to address healthcare accessibility challenges in Morocco. The application leverages AI to provide medical guidance, pre-diagnosis, and medication information in both Moroccan Arabic (Darija) and English.

## Project Structure

The project consists of three main implementation variations:
- `main.py` - Core implementation using Google's Generative AI
- `main-llama.py` - Implementation using Llama model
- `main-multimodel.py` - Enhanced implementation supporting multiple AI models

## Key Features

1. **Medical Pre-Diagnosis**
   - Symptom analysis and preliminary health assessment
   - Urgency evaluation for medical conditions
   - Available in English with Darija support coming soon

2. **Medication Information**
   - Extract medication details via OCR from uploaded images
   - Provide usage instructions and potential side effects
   - Search medication information by name

3. **Multi-language Support**
   - English and Moroccan Arabic (Darija) interfaces
   - Translation capabilities between languages

4. **User Privacy Protection**
   - Option to modify personal information before submission
   - Control over sensitive medical data

## Background and Problem Statement

In Morocco, many people struggle to access healthcare or receive timely advice due to language barriers, lack of resources, or the unavailability of nearby medical professionals. Specifically, Moroccan users face challenges when trying to use large language models or medical applications for self-diagnosis and understanding their health conditions. Additionally, many patients fail to fully read or understand the information provided with medications, leading to potential misuse or missed side effects. Furthermore, patients often do not know whether a medical issue is urgent or not.

The lack of accessibility to reliable health resources in Moroccan Arabic (Darija) compounds the problem, as most available applications and medical information are in English or other languages that are not native to many Moroccans. This can prevent them from making informed decisions about their health, and hinder their ability to manage their conditions appropriately.

## Team Members

- Anas Ouhannou
- Belkhlifi Anas
- Imane El Maakoul
- Mayssae Jabbar
- Wajih Esghayri

## Impact and Proposed Solution

This project aims to solve these problems by providing a web application that allows Moroccan users to interact with a chatbot for medical assistance and pre-diagnosis. The application supports both Darija (Moroccan Arabic) and English, making it more accessible to the local population.

The application enables users to:
- **Pre-diagnose health conditions** based on symptoms and medical history (available in English, with Darija support coming soon).
- **Get information about medications**, including potential side effects and usage instructions, by either uploading an image of the medication or entering its name.
- **Ensure confidentiality** by allowing users to input and modify their personal information (age, gender, symptoms, medical history) and delete any sensitive information before submission.

By offering accurate and reliable medical guidance in Darija and English, the app helps Moroccan users make informed decisions about their health, and reduces the risk of misunderstanding medication instructions.

## Technical Implementation

### Components and Technologies

1. **Frontend**: Built with Streamlit for a responsive and interactive user interface
2. **OCR Capabilities**: Using pytesseract to extract text from medication images
3. **AI Models**:
   - Google Generative AI (main.py)
   - Llama Model (main-llama.py)
   - Multi-model support (main-multimodel.py)
4. **Translation**: Using googletrans and deep_translator libraries for language conversion
5. **Image Processing**: Utilizing PIL for image manipulation and preparation for OCR

### Main Architecture

The application follows a streamlined architecture:
1. User inputs (text, images) → 
2. Processing (OCR, translation) → 
3. AI model inference → 
4. Response generation → 
5. User-friendly display

## Requirements and Setup

### Dependencies

```
streamlit
pytesseract
Pillow
requests
python-dotenv
google-generativeai>=0.8.0
transformers (for main-llama.py and main-multimodel.py)
llama_cpp (for main-llama.py and main-multimodel.py)
googletrans==4.0.0-rc1 (for main-llama.py and main-multimodel.py)
deep_translator (for main-llama.py and main-multimodel.py)
```

### Prerequisites for llama_cpp
The llama_cpp library requires a C++ compiler and additional tools for building C++ extensions:

1. **C++ Compiler**:
   - Windows: Install Visual Studio Build Tools
   - macOS: Install Xcode Command Line Tools (`xcode-select --install`)
   - Linux: Install GCC (`sudo apt-get install build-essential`)

2. **CMake**:
   - Windows: Download from the CMake website
   - macOS: `brew install cmake`
   - Linux: `sudo apt install cmake`

### Tesseract OCR Installation
Required for image text extraction:
- Windows: Download Tesseract OCR
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

## Running the Application

To start the application, navigate to the project directory and run:

```bash
# For the standard version
streamlit run main.py

# For the Llama model version
streamlit run main-llama.py

# For the multi-model version
streamlit run main-multimodel.py
```

## Project Status and Future Work

The application is currently functional with core features implemented. Future work includes:
- Enhanced Darija language support for medical pre-diagnosis
- Expanded medication database
- Integration with healthcare provider systems
- Mobile application development

## Contact Information

For questions or issues with the project, please contact:
- Email: [anasouhnou@gmail.com](mailto:anasouhnou@gmail.com)

---

**Note**: This application is intended for informational purposes only and does not replace professional medical advice. Always consult with a healthcare provider for serious medical concerns.
