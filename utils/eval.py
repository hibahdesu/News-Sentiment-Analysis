from sklearn.metrics import classification_report, confusion_matrix

def eval(model, X_train, X_test, y_train, y_test):
    # Generate predictions for both train and test sets
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Print confusion matrix for the test set
    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))

    # Print classification report for the test set
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # Print classification report for the training set
    print("\nClassification Report (Train Set):")
    print(classification_report(y_train, y_pred_train))

