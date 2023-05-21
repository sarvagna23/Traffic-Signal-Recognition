import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# Load the trained model to classify signs
model = load_model('traffif_sign1_model.h5')

class_names = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
    'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
    'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
    'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
    'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

root = tk.Tk()

def grayscale(img):
    return img.convert("L")

def equalize(img):
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = np.array(img) / 255
    return img

def getClassName(classNo):
    if classNo >= 0 and classNo < len(class_names):
        return class_names[classNo]
    else:
        return 'Unknown'


def classify_sign(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 1)
    img = img / 255.0

    result = model.predict(img)
    class_index = np.argmax(result)
    confidence = result[0][class_index]

    return class_names[class_index], confidence

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img

        className, confidence = classify_sign(file_path)
        text_label.config(text=f"Class: {className}\nConfidence: {confidence}%")

        # Text-to-speech output
        tts = gTTS(text=className, lang='en')
        tts.save('output.mp3')
        audio = AudioSegment.from_file("output.mp3", format="mp3")
        play(audio)

label = tk.Label(root)
label.pack()

browse_button = tk.Button(root, text="Browse", command=browse_image)
browse_button.pack()

text_label = tk.Label(root, text="")
text_label.pack()

root.mainloop()