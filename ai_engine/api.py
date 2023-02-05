from flask import Flask
from feature_extractor import ExtractFeatures
import run_model as rm
from model_training import MLP

app = Flask(__name__)


@app.route('/<path:text>', methods=['GET', 'POST'])
def run_model(text):
    url = text
    print(f'URL: {url}')
    features = ExtractFeatures().extract_features(url)
    print(f'Features: {features}')
    result = rm.run_model(features)
    print(f'Result: {result}')
    
    if float(result['confidence']) > 0.70:
        return 'pass'
    else:
        return 'not_pass'


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
