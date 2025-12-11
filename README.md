Multilingual Language Detection Using LSTM

A deep learning–based language identification system that classifies text into English, Kannada, Hindi, or Tamil using a Character-Level Long Short-Term Memory (LSTM) neural network.
This project demonstrates how sequential deep learning models can effectively detect languages by analyzing character patterns instead of relying on word-level features.

🚀 Project Overview

Multilingual language detection is an essential component in modern NLP applications such as:

Chatbots

Social media analytics

Translation engines

Information retrieval systems

Content moderation

Traditional rule-based or word-based methods often fail for short texts, informal writing, or mixed scripts.
This project solves that using an LSTM model, which learns script- and sequence-based patterns at the character level.

📌 Features

✔ Character-level tokenizer for script-based language representation
✔ Custom multilingual dataset (Wikipedia-sourced + augmented English data)
✔ LSTM neural network built using TensorFlow/Keras
✔ High accuracy on short and noisy text inputs
✔ Visualizations: accuracy & loss curves + confusion matrix
✔ Supports prediction for real-time text input

📁 Dataset

The dataset is built using:

Wikipedia sentences in English, Kannada, Hindi, and Tamil

Additional English samples for class balancing

Cleaned, tokenized, padded text sequences

Each row contains:

text	language
"भारत एक विशाल देश है।"	Hindi
"Welcome to the world of AI."	English
🧠 Model Architecture

The LSTM model includes:

Embedding Layer – character-level vector representation

LSTM Layer (128 units) – learns sequential dependencies

Dropout Layer (0.3) – prevents overfitting

Dense Layer – classification using Softmax

Loss: categorical_crossentropy
Optimizer: Adam
Metrics: Accuracy

🔧 Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Scikit-learn

Matplotlib

Jupyter Notebook / Google Colab

⚙️ How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/multilingual-language-detection-lstm.git
cd multilingual-language-detection-lstm

2. Install dependencies
pip install -r requirements.txt

3. Run the training script
python train_model.py

4. Test the model
python predict.py

📊 Results

The model is evaluated using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Training/Validation Accuracy graph

Training/Validation Loss graph

The LSTM model shows strong performance in distinguishing between four languages even for short sequences.



📝 Sample Usage
from language_identifier import predict_language

text = "ನೀವು ಹೇಗಿದ್ದೀರಿ?"
lang = predict_language(text)
print("Predicted Language:", lang)


Project Structure:
MULTILINGUAL/
│
├── app.py                         # (Optional) Script for running the prediction interface or API
├── ex.py                          # Dataset collection script (Wikipedia scraping)
├── language_identifier.py          # Main script to load model & predict language
├── tempCodeRunnerFile.py           # Temporary VS Code runner file (auto-generated)
│
├── extra_english.csv               # Additional English dataset used for augmentation
├── mini_multilingual.csv           # Base multilingual dataset from Wikipedia
├── mini_multilingual_aug.csv       # Combined + augmented dataset
├── input.txt                       # Sample input text file for testing
│
├── language_model.h5               # Trained LSTM language detection model
├── tokenizer.pkl                   # Saved tokenizer (character-level)
├── label_encoder.pkl               # Saved label encoder (maps classes to indices)
│
└── README.md (recommended) 

Output:

Predicted Language: Kannada

📌 Future Enhancements

Add more regional + global languages

Use GRU or Transformer-based architecture

Deploy as a REST API or Streamlit web app

Create a mobile-compatible prediction interface
