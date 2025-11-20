# AI-ML-in-Drug-Discovery
AI-ML in Drug Discovery
# QSAR Modeling for Aqueous Solubility Prediction (LogS) 🧪🔬

## Overview

This project implements **Quantitative Structure-Activity Relationship (QSAR)** modeling using machine learning techniques to predict the **Aqueous Solubility (LogS)** of small molecules. Aqueous solubility is a critical physicochemical property in drug discovery, influencing absorption, distribution, metabolism, and excretion (ADME) characteristics.

The model uses four key molecular descriptors as features:
1. **MolLogP**: Octanol-water partition coefficient (a measure of lipophilicity).
2. **MolWt**: Molecular Weight.
3. **NumRotatableBonds**: Number of rotatable bonds.
4. **AromaticProportion**: Proportion of heavy atoms in aromatic rings.

Four different regression algorithms were trained and evaluated: **Linear Regression**, **Random Forest**, **Support Vector Regression (SVR)**, and **Gradient Boosting Regressor (GBR)**. The best-performing model (GBR) was then optimized using **GridSearchCV** for hyperparameter tuning.

---

## Project Structure

* `AI-ML in Drug Discovery.ipynb`: The Jupyter notebook containing all the data preparation, model training, evaluation, and tuning steps.
* `solubility_with_descriptors.csv`: The dataset used for training and testing (containing LogS values and molecular descriptors).
* `tuned_gbr_model.joblib`: The final, best-performing Gradient Boosting Regressor model saved after hyperparameter tuning.

---

## Data

The dataset consists of molecular descriptors as **features (X)** and the corresponding experimental aqueous solubility (`logS`) as the **target variable (y)**.

| Feature Name | Description |
| :--- | :--- |
| **MolLogP** | Molecular LogP (lipophilicity) |
| **MolWt** | Molecular Weight (g/mol) |
| **NumRotatableBonds** | Number of rotatable bonds |
| **AromaticProportion** | Proportion of aromatic atoms |
| **logS** | Experimental Log-transformed Solubility (target) |

---

## Methodology and Results

The dataset was split into 80% for training and 20% for testing. All models were evaluated using **Mean Squared Error (MSE)** and **R-squared ($R^2$)** score.

### 1. Model Comparison and Evaluation

The Gradient Boosting Regressor performed the best among the untuned models. Subsequent hyperparameter tuning further refined its performance.

| Method | Training MSE | Training $R^2$ | Test MSE | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| Linear regression | 1.0075 | 0.7645 | 1.0207 | 0.7892 |
| Random forest | 1.0282 | 0.7597 | 1.4077 | 0.7092 |
| Support Vector Regression | 2.3290 | 0.4556 | 2.7211 | 0.4379 |
| **Gradient Boosting Regressor (Untuned)** | 0.3374 | 0.9211 | 0.6707 | 0.8615 |
| **Tuned Gradient Boosting Regressor** | 0.6308 | 0.8697 | **0.6308** | **0.8697** |

### 2. Hyperparameter Tuning (GBR)

The GBR model was tuned using `GridSearchCV` over a defined parameter space:

| Parameter | Values Tested |
| :--- | :--- |
| `n_estimators` | [100, 200, 300] |
| `learning_rate` | [0.01, 0.1, 0.2] |
| `max_depth` | [3, 4, 5] |
| `min_samples_split` | [2, 5, 10] |

**Best Parameters Found:**

{'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}

## Overall Conclusion

The **Tuned Gradient Boosting Regressor** is the superior model for predicting aqueous solubility (LogS) based on the provided molecular descriptors. It achieved the best overall predictive accuracy on the unseen test data, with a final R-squared value of **0.8697** and a low Mean Squared Error of **0.6308**. This demonstrates a strong and reliable correlation between the calculated molecular descriptors and the experimental LogS values.

---

## Usage

To reproduce the results or use the saved model:

1. **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2. **Install the necessary libraries:**
    ```bash
    pip install pandas scikit-learn matplotlib plotly joblib
    ```
3. **Run the Jupyter Notebook:**
    Execute the cells in `AI-ML in Drug Discovery.ipynb` to see the full analysis, training process, and visualizations.
4. **Load the final model:**
    You can load the saved, tuned GBR model for new predictions using `joblib`:
    ```python
    import joblib
    import pandas as pd
    
    # Load the model
    best_gbr = joblib.load('tuned_gbr_model.joblib')
    
    # Example prediction: Prepare new data with the 4 required descriptors
    # Note: Column names must match: 'MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'
    # new_data = pd.DataFrame({'MolLogP': [X.X], 'MolWt': [Y.Y], ...})
    # prediction = best_gbr.predict(new_data)
    ```
