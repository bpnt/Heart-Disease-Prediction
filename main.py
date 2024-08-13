from data_loader import load_data
from preprocess import preprocess_data, standardize_data
from models import get_models
from evaluation import train_model

def main():
    df = load_data('heart.csv')
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train, X_test = standardize_data(X_train, X_test)

    models = get_models()

    for model in models:
        train_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
