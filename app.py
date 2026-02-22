from flask import Flask, request, render_template, jsonify
import pickle, re, sys, time
sys.path.insert(0, 'src')
from preprocess import preprocess, extract_features

app = Flask(__name__)
model = pickle.load(open('model/spam_model.pkl', 'rb'))
vec   = pickle.load(open('model/vectorizer.pkl', 'rb'))

SPAM_TRIGGERS = [
    'free','win','winner','cash','prize','click','urgent',
    'congratulations','offer','limited','guaranteed','deal',
    'buy','cheap','discount','claim','selected','billion'
]

def analyze(text):
    clean    = preprocess(text)
    features = extract_features(text)
    X        = vec.transform([clean])
    pred     = model.predict(X)[0]
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[0][1]
    elif hasattr(model, 'decision_function'):
        import numpy as np
        raw  = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-raw))
    else:
        prob = float(pred)
    words_found = [w for w in SPAM_TRIGGERS if re.search(rf'\b{w}\b', text, re.I)]
    return {
        'label':      'SPAM' if pred == 1 else 'HAM',
        'confidence': round(float(prob) * 100, 1),
        'is_spam':    bool(pred),
        'triggers':   words_found,
        'features':   features,
        'word_count': len(text.split()),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    time.sleep(0.4)
    return jsonify(analyze(text))

if __name__ == '__main__':
    app.run(debug=True)