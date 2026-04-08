JIGSAW_COMPONENTS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

PERSPECTIVE_API_KEY = ""
PERSPECTIVE_URL = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}"
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

CACHE_DIR = "/root/autodl-tmp"
HF_TOKEN = ""