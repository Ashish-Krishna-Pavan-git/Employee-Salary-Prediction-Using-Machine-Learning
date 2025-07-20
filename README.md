# Employee Salary Prediction & Interactive Web Application

## 📌 Project Overview

This project focuses on building a robust Machine Learning model to predict an individual's income level based on various demographic and employment-related features. The solution includes an interactive Streamlit web application that allows users to input characteristics and receive real-time salary predictions.

---

## ✨ Features

- **Data Preprocessing:** Handles missing values, cleans data, and prepares features for model training.
- **Feature Engineering:** Introduces insightful features like `net_capital` from existing attributes.
- **Multiple ML Models:** Compares performance across Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and MLP Classifier.
- **Optimal Model Selection:** Chooses the best-performing model using **F1 Score**, ideal for handling class imbalance.
- **Interactive Web App:** User-friendly Streamlit interface for live predictions and model performance visualization.
- **Model Persistence:** Trained models are saved and loaded seamlessly for web deployment.

---

## 📂 Dataset

The project utilizes the **Adult Census Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult), containing demographic and income information.  
The target variable classifies income as `>50K` or `<=50K`.  
A key challenge addressed was the significant **class imbalance** in the dataset.

---
Caution: Do not run all cells in notebook because last cell needed to to be stopped manually, to using the application running first nine cells are enough for application to run to stop application teriminate cell 9 and execute cell 10
---
## 🧱 Project Structure

```plaintext
/Employee Salary Prediction Application/
├── app.py                          # Streamlit web application
├── Assets/
│   ├── model_performance.csv       # Model evaluation metrics
│   ├── roc_data.json               # (Optional) ROC curve data
│   └── X_train_sample.csv          # (Optional) Sample training data for input
├── Employee_Dataset.csv            # Raw dataset
├── Final Project Presentation.pptx # Presentation slides
├── Models/
│   ├── logistic_regression_pipeline.pkl
│   ├── mlp_classifier_pipeline.pkl
│   └── xgboost_pipeline.pkl        # Trained model pipelines
├── salary_prediction_model.ipynb  # Jupyter notebook for data processing + training
├── README.md                       # Project documentation
└── LICENSE                         # Custom license terms
```

---

## ⚙️ Setup and Installation

To set up and run this project locally:

1. **Clone the Repository (or Download Files):**
   ```bash
   git clone <your-repository-url>
   cd "Employee Salary Prediction Application"
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```
   _Or install manually:_
   ```bash
   pip install pandas matplotlib seaborn scikit-learn xgboost streamlit streamlit-option-menu psutil joblib numpy
   ```

---

## 🚀 How to Run the Application

1. **Run the Jupyter Notebook:**
--
##
***Alert: Do not run all cells because last cell needed to to be stopped manually after using the application***

   
   Open `salary_prediction_model.ipynb` in Jupyter Notebook or VS Code and run all cells.  
   This will:
   - Clean and preprocess data
   - Train models
   - Save pipelines to `Models/`
   - Generate model evaluation CSV in `Assets/`

3. **Launch the Streamlit Web App:**
   In your terminal, run:
   ```bash
   streamlit run app.py
   ```
   This will open the app in your default web browser.

---

## 📊 Model Performance

Models were evaluated using **Accuracy**, **F1 Score**, and **ROC AUC**.  
**XGBoost** emerged as the best-performing model based on **F1 Score**, a key metric for imbalanced classification tasks.

Detailed metrics are available in:

- `Assets/model_performance.csv`
- Interactive charts inside the Streamlit app

---

## 📝 License

This project is distributed under a **Custom License**.

- ✅ Free for **personal, academic, or non-commercial** use.
- ❌ **Commercial use, redistribution, sublicensing, or claiming credit** is **not allowed** without prior written permission.

For any commercial or redistribution requests, contact:

📧 **ashishkrishnapavan@gmail.com**

---

## 📬 Contact

For queries, feedback, or collaboration opportunities:

**Ashish Krishna Pavan**  
📧 Email: [ashishkrishnapavan@gmail.com](mailto:ashishkrishnapavan@gmail.com)
