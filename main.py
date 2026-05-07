import streamlit as st
import speech_recognition as sr
import pickle
import numpy as np
from tensorflow.keras.models import load_model

import webbrowser
import os
import platform

model = load_model("model.h5")

with open("vectorizer.pkl" , "rb") as f:
    vectorizer = pickle.load(f)
    
with open("label_encoder.pkl" , "rb") as f:
    encoder = pickle.load(f)
    
st.set_page_config(page_title = "Voice ANN Classifier or AI Asisstant" , layout = "wide" )
st.title("AI Assisstant")
st.write("Click the button and speak a command")

def perform_action(command):
    if command == "browser":
        webbrowser.open("https://www.google.com")
        return "Opening the Browser"
    elif command == "music":
        webbrowser.open("https://soundcloud.com")
        
        return "Playing the Music"
    elif command == "youtube":
        webbrowser.open("https://www.youtube.com/watch?v=NlfynDqK7r8&list=RDNlfynDqK7r8&start_radio=1")
        return "Opening the Youtube and Playing the song"
    elif command == "notepad":
        if platform.system() == "Windows":
            os.system("notepad")
            return "Opening the Notepad"
        else:
            return "Notepad Supoorted only on Windows"
    elif command == "shutdown":
        return "Shutdown command is disabled for Safety Purpose"
    else:
        return "No action Defined"

def recognize_speech():
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening.....")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Could not understand the Voice"
    except sr.RequestError:
        return "API Unavailable"
    except Exception as e:
        return f"Error : {e}"

if st.button("🎤 Speak Now!"):
    user_text = recognize_speech()
    st.success(f"You Said: {user_text}")
    
    if "❌" not in user_text:
        X = vectorizer.transform([user_text]).toarray()
        
        prediction = model.predict(X)
        label_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([label_index])[0]
        
        st.subheader(f"Prediction: {predicted_label}")
        
        result = perform_action(predicted_label)
        st.success(result)