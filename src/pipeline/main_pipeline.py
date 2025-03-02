import sys
import os

# Add both src and utils to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from src.pipeline.load_data import load_data
from src.pipeline.clean_data import clean_data
from src.pipeline.process_text import process_text
from src.pipeline.encode_sentiment import encode_sentiment
from src.pipeline.split_data import split_data
from src.pipeline.vectorize_data import vectorize_data
from src.pipeline.train_logistic_model import train_logistic_model
from src.pipeline.train_svc_model import train_svc_model
from src.pipeline.save_models import save_models
from utils.eval import eval

def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    df = process_text(df)
    
    # Encode sentiment
    df, encoder = encode_sentiment(df)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Vectorize the data
    X_train_tf, X_test_tf, vectorizer = vectorize_data(X_train, X_test)
    
    # Train models
    log_model = train_logistic_model(X_train_tf, y_train)
    svc_model = train_svc_model(X_train_tf, y_train)
    
    # Evaluate models
    print("Logistic Regression Model Evaluation")
    eval(log_model, X_train_tf, X_test_tf, y_train, y_test)  # Pass y_train and y_test
    
    print("SVC Model Evaluation")
    eval(svc_model, X_train_tf, X_test_tf, y_train, y_test)  # Pass y_train and y_test
    
    # Save the best model (For demonstration, we'll save the logistic model)
    save_models(log_model, vectorizer, 'logistic_model')
    
    print("Best model and vectorizer saved.")

if __name__ == '__main__':
    main()
