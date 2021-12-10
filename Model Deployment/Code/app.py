import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
import re
from transformers import DistilBertTokenizerFast,TFDistilBertForSequenceClassification
# Define a flask app
app = Flask(__name__)

MODEL_PATH = "model/"

model=TFDistilBertForSequenceClassification.from_pretrained("model/")
print(f'model: {model}')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Remove symbols from text and lower case all characters
def process_text(text):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",text[0].lower()).split())

def model_predict(text,model,tokenizer):
    # Encode and predict
    text_encodings = tokenizer(text, truncation=True, padding=True)
    print(text_encodings)
    preds = model.predict(tf.data.Dataset.from_tensor_slices(dict(text_encodings))).logits
    preds = tf.keras.activations.softmax(tf.convert_to_tensor(preds)).numpy()
    print(f"Preds {preds}")
    return preds

  
@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        text = request.form['text1']
        text =[''.join(c for c in text)]
        text = process_text(text)
        print([text])
        # Make prediction
        preds = model_predict([text], model, tokenizer)
        print("Success!")
        # Process the result
        max_index = np.where(preds == np.amax(preds[1:-1]))[0][0]
        preds=preds[max_index]
        print(preds)
        pred_class = preds.argmax(axis=-1)
        pred_class = str(pred_class) # Convert to string
        print(pred_class)
        return pred_class
    return None

if __name__ == '__main__':            
    app.run(debug=True)