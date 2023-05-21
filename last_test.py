import numpy as np
import cv2
from tensorflow import keras
from gtts import gTTS
import os
from tkinter import *
from tkinter import filedialog, messagebox
import tkinter as tk
import train
from pydub import AudioSegment
from pydub.playback import play

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        
        # Display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(1, 32, 32, 1)
        img = img / 255.0

        # Make prediction
        result = model.predict(img)
        class_index = np.argmax(result)
        confidence = result[0][class_index]
        class_name = train.class_names[class_index]

        # Text-to-speech output
        tts = gTTS(text=class_name, lang='en')
        tts.save('output.mp3')
        
        # Display dialog box
        messagebox.showinfo("Sign Recognition", f"Class: {class_name}\nConfidence: {confidence}")
        
        audio = AudioSegment.from_file("output.mp3", format="mp3")
        play(audio)

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == 13:  # Press Enter key to capture image
            cv2.imwrite('captured_image.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Load and display the captured image
    img = cv2.imread('captured_image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    cv2.imshow("Captured Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 1)
    img = img / 255.0
    
    # Make prediction
    result = model.predict(img)
    class_index = np.argmax(result)
    confidence = result[0][class_index]
    class_name = train.class_names[class_index]
    
    # Display the result in a dialog box
    messagebox.showinfo("Sign Recognition", f"Class: {class_name}\nConfidence: {confidence}")
    
    tts = gTTS(text=class_name, lang='en')
    tts.save('output.mp3')
    audio = AudioSegment.from_file("output.mp3", format="mp3")
    play(audio)

root = Tk()
root.title("OUTPUT")
root.geometry("2500x1000")
root.configure(bg="white")
label1 = Label(root, text="TRAFFIC SIGN RECOGNITION AND ALERTING SYSTEM", font=(
    "Sans-serif", 40, "bold"), bg="white", fg="black", width=1500, height=200)
label1.pack()
threshold = 0.97 # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffif_sign1_model.h5') # load model
btn1 = Button(root, text="USE CAMERA", font=("Sans-serif", 10, "bold"),
bg="black", fg="white", width=50, height=50, command=capture_image)
btn1.place(x=200, y=200)
btn2 = Button(root, text="BROWSE IMAGE", font=("Sans-serif", 10, "bold"),
bg="black", fg="white", width=50, height=50, command=browse_image)
btn2.place(x=400, y=200)
root.mainloop()