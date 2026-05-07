import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = {
    "text": [
        "open browser" , "start browser" , "chrome open kren",
        "play music" , "start music" , "dj walay babu mera gana chala du",
        "shutdown system" , "turn off computer" , "power off system",
        "open notepad" , "start notepad" , "launch notepad" ,
        "open youtube" , "play youtube" , "start youtube"
    ],
    "label" : [
        "browser" , "browser" , "browser" ,
        "music" , "music" , "music",
        "shutdown" , "shutdown" , "shutdown",
        "notepad" , "notepad" , "notepad",
        "youtube" , "youtube" , "youtube"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"]).toarray()
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

model = Sequential([
    Dense(32, activation="relu" , input_dim = X.shape[1]),   #Input Layer
    Dense(16, activation = "relu"),    #Hidden Layer
    Dense(len(set(y)), activation = "softmax")  
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.fit(X, y, epochs = 150, verbose = 1)

model.save("model.h5")

with open("vectorizer.pkl" , "wb") as f:
    pickle.dump(vectorizer, f)
    
with open("label_encoder.pkl" , "wb") as f:
    pickle.dump(encoder, f)

print("Model Trained with commands")