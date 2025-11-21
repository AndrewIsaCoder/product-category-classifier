# Product Category Prediction

This project implements a machine learning pipeline to predict product categories based on their titles. It includes data analysis, a training script, and an interactive prediction tool.

## Project Structure

```
├── data/
│   └── products.csv       # Dataset (Synthetic or Real)
├── models/
│   └── model.pkl          # Trained model pipeline
├── notebooks/
│   └── analysis.ipynb     # Jupyter notebook for EDA and experiments
├── src/
│   ├── train_model.py     # Script to train and save the model
│   └── predict_category.py # Script for interactive prediction
└── README.md              # Project documentation
```

## Setup

1. **Clone the repository** (if applicable).
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. **Prepare Data**:
   - Place your `products.csv` file in the `data/` directory.
   - **Note**: A synthetic dataset is provided for demonstration. For real-world performance, replace it with the actual dataset.

## Usage

### 1. Train the Model

Run the training script to process the data and save the model:

```bash
cd src
python train_model.py
```

This will create `models/model.pkl`.

### 2. Predict Categories

Run the interactive prediction script:

```bash
cd src
python predict_category.py
```

Enter a product title when prompted (e.g., "iPhone 13 Pro") to see the predicted category.

### 3. Explore Analysis

Open `notebooks/analysis.ipynb` in Jupyter Notebook or VS Code to view the exploratory data analysis and model comparison results.

## Model Details

- **Algorithm**: The pipeline compares Naive Bayes, Linear SVC, and Random Forest. The best performing model is selected automatically.
- **Features**: TF-IDF vectorization of the product title.
- **Metrics**: Accuracy, Precision, Recall, F1-Score.

## Author

Stoian Andrei <3 - Link Academy final project
