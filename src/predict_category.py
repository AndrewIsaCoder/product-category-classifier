import pickle
import os
import sys

# Configuration
MODEL_PATH = os.path.join('models/model.pkl')

def load_model(path):
    """Load the trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Please run train_model.py first.")
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_category(model, title):
    """Predict the category for a given product title."""
    prediction = model.predict([title])[0]
    return prediction

def main():
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        print("Enter a product title to predict its category (or 'exit' to quit).")
        
        while True:
            user_input = input("\nProduct Title: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            if not user_input:
                continue
                
            category = predict_category(model, user_input)
            print(f"Predicted Category: {category}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
