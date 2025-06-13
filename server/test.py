# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle
# from tensorflow.keras.preprocessing.text import Tokenizer
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the model (chat_model.h5)
# model = load_model('chat_model.h5')

# # Print the model architecture
# print("Model Architecture:")
# model.summary()  # Displays a summary of the model

# # Optionally: View weights of specific layers (example)
# layer_name = 'dense'  # Replace with your layer name
# weights = model.get_layer(layer_name).get_weights()
# print(f"\nWeights of layer {layer_name}: {weights}")

# # Load the tokenizer (tokenizer.pickle)
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Print the tokenizer's word index (vocabulary)
# print("\nTokenizer Word Index:")
# print(tokenizer.word_index)

# # Load the label encoder (label_encoder.pickle)
# with open('label_encoder.pickle', 'rb') as ecn_file:
#     lbl_encoder = pickle.load(ecn_file)

# # Print the label encoder's classes (unique intents)
# print("\nLabel Encoder Classes:")
# print(lbl_encoder.classes_)

# # Example: Test model prediction on a sample input sentence
# sample_sentence = "Hello, how can I help you?"
# sample_sequence = tokenizer.texts_to_sequences([sample_sentence])  # Tokenize the sentence
# max_len = 20  # Use the same max_len used during training
# padded_sample = tf.keras.preprocessing.sequence.pad_sequences(sample_sequence, truncating='post', maxlen=max_len)  # Pad the sequence

# # Predict the intent
# prediction = model.predict(padded_sample)
# predicted_class = np.argmax(prediction, axis=1)
# predicted_intent = lbl_encoder.inverse_transform(predicted_class)

# print(f"\nPredicted Intent for '{sample_sentence}': {predicted_intent[0]}")

# # Optionally: Plot loss and accuracy curves if you have the training history
# # Assuming `history` is available from the training process
# # Uncomment the following lines if you have the `history` object from training
# # plt.figure(figsize=(12, 5))

# # Accuracy plot
# # plt.subplot(1, 2, 1)
# # plt.plot(history.history['accuracy'], label='Training Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # plt.title('Accuracy Over Epochs')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()

# # Loss plot
# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['loss'], label='Training Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.title('Loss Over Epochs')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()

# # plt.tight_layout()
# # plt.show()


import json
import numpy as np
import re
import pickle
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize spaCy's English tokenizer
nlp = spacy.load('en_core_web_sm')

# Function to preprocess and lemmatize text (same as during training)
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# Function to convert text to sequence and pad it
def prepare_input(text, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=50, padding='post')

# Load the trained model
model = load_model('chat_model.h5')

# Load the tokenizer and label encoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)


# OR, if you don't have the validation data stored, you need to re-create the train-test split
# Example: (assuming X and y are the input features and labels, respectively)

# Load or define X and y (from intents.json, for example)
with open('intents.json') as file:
    intents = json.load(file)

patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = preprocess_text(pattern)
        patterns.append(' '.join(word_list))  # Add tokenized pattern
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Convert patterns to sequences
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X, maxlen=50, padding='post')

# Encode the tags
y = label_encoder.transform(tags)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy score
accuracy = accuracy_score(y_val, y_pred_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_val, y_pred_classes, average='weighted')
recall = recall_score(y_val, y_pred_classes, average='weighted')
f1 = f1_score(y_val, y_pred_classes, average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
report = classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)

# Plotting training and validation loss and accuracy from the history of the model
history = model.history  # This will only work if you return the history during training (model.fit returns it)

# Plot Training & Validation Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
