from sklearn.metrics import classification_report, accuracy_score

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))