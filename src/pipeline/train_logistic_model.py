from sklearn.linear_model import LogisticRegression

def train_logistic_model(X_train_tf, y_train):
    log_model = LogisticRegression(C=0.4, max_iter=1000, class_weight='balanced')
    log_model.fit(X_train_tf, y_train)
    return log_model
