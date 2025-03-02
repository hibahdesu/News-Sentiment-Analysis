from sklearn.svm import LinearSVC

def train_svc_model(X_train_tf, y_train):
    svc_model = LinearSVC(C=0.1, class_weight='balanced')
    svc_model.fit(X_train_tf, y_train)
    return svc_model
