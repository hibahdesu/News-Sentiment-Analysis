import joblib

def save_models(model, vectorizer, model_name):
    joblib.dump(model, f'saved_models/{model_name}.pkl')
    joblib.dump(vectorizer, 'saved_models/vectorizer.pkl')
