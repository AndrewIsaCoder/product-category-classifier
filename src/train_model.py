import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# Căi către fișiere (presupunem că rulezi scriptul din root-ul proiectului)
DATA_PATH = os.path.join("data", "products.csv")
MODEL_PATH = os.path.join("models", "model.pkl")

# Nume coloane din CSV – SCHIMBĂ aici dacă în fișier sunt altfel!
TITLE_COL = "Product Title"
CATEGORY_COL = " Category Label"


# --------------------------------------------------
# FUNCȚII UTILE
# --------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV and afișează coloanele găsite."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    print("Columns found in dataset:")
    print(list(df.columns))

    # verificăm că există coloanele necesare
    missing_cols = [col for col in [TITLE_COL, CATEGORY_COL] if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing expected columns: {missing_cols}. "
            f"Please check the CSV header and update TITLE_COL / CATEGORY_COL in train_model.py."
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: drop NaN, duplicates, normalize text."""
    print("\nCleaning data...")

    # păstrăm doar rândurile care au titlu și categorie
    df = df.dropna(subset=[TITLE_COL, CATEGORY_COL])

    # eliminăm duplicatele pe titlu (opțional, dar util)
    df = df.drop_duplicates(subset=[TITLE_COL])

    # normalizăm titlul: string + lowercase
    df[TITLE_COL] = df[TITLE_COL].astype(str).str.lower()

    print(f"Number of rows after cleaning: {len(df)}")
    return df


def train_model(df: pd.DataFrame):
    """Train and evaluate multiple models, return the best one."""
    print("\nPreparing data for training...")

    X = df[TITLE_COL]
    y = df[CATEGORY_COL]

    # împărțim în train/test (stratify pentru distribuție similară a categoriilor)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # definim modelele sub formă de pipeline (TF-IDF + classifier)
    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB()),
        ]),
        "Linear SVC": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LinearSVC()),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
        ]),
    }

    best_model = None
    best_model_name = None
    best_score = -1.0

    for name, pipeline in models.items():
        print(f"\n==============================")
        print(f"Training model: {name}")
        print(f"==============================")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {score:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name

    print("\n======================================")
    print(f"Best model: {best_model_name} with accuracy {best_score:.4f}")
    print("======================================")

    return best_model


def save_model(model, path: str):
    """Save the trained model (pipeline) to disk as .pkl."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Starting training pipeline...")

    try:
        df = load_data(DATA_PATH)
        df = clean_data(df)
        model = train_model(df)
        save_model(model, MODEL_PATH)
        print("\nTraining complete.")
    except Exception as e:
        print("\nERROR during training:")
        print(e)


if __name__ == "__main__":
    main()
