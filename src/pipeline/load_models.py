import joblib

def load_saved_model(model_name):
    model = joblib.load(f'saved_models/{model_name}.pkl')
    vectorizer = joblib.load('saved_models/vectorizer.pkl')
    return model, vectorizer
