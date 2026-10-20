# 🏦 Loan Approval Prediction Using Decision Tree Classifier

## 📌 Project Overview

This project focuses on predicting loan approval status using a **Decision Tree Classifier**, a supervised machine learning algorithm. The model analyzes applicant information, including income, credit history, education, marital status, loan amount, and property area, to predict whether a loan application is likely to be approved or rejected.

Financial institutions receive a large number of loan applications and need effective ways to evaluate applicants. Machine learning can help identify patterns in historical application data and support more consistent, data-driven assessments.

In this project, I perform data preprocessing, exploratory data analysis (EDA), feature encoding, model training, evaluation, and decision tree visualization using Python and its machine learning libraries.

## 🎯 Problem Statement

The objective of this project is to develop a machine learning classification model that predicts whether a loan application will be approved or rejected based on the applicant's financial and demographic information.

The project uses historical loan application data to identify relationships between applicant characteristics and loan approval outcomes. A Decision Tree Classifier is trained on the processed dataset and evaluated on unseen test data to assess its predictive performance.

The project also visualizes the trained decision tree and examines feature importance to better understand the factors influencing the model's predictions.

## 🎯 Project Objectives

- Understand and explore the loan application dataset.
- Identify and handle missing values in the dataset.
- Convert categorical variables into numerical representations.
- Perform exploratory data analysis to understand feature distributions and relationships.
- Analyze correlations between the input features and loan approval status.
- Split the dataset into training and testing sets.
- Build a Decision Tree Classification model.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Visualize the trained decision tree and examine feature importance.
- Generate predictions and class probabilities for sample loan applicants.

## 📊 Dataset Description

The project uses a loan prediction training dataset named `train_u6lujuX_CVtuZ9i (1).csv`.

The dataset contains information about loan applicants and a target variable representing whether their applications were approved.

### Features used in the model

| Feature | Description |
|---|---|
| `Gender` | Gender of the applicant |
| `Married` | Marital status of the applicant |
| `Dependents` | Number of dependents |
| `Education` | Applicant's education level |
| `Self_Employed` | Whether the applicant is self-employed |
| `ApplicantIncome` | Income of the applicant |
| `CoapplicantIncome` | Income of the co-applicant |
| `LoanAmount` | Requested loan amount |
| `Loan_Amount_Term` | Loan repayment term |
| `Credit_History` | Credit history indicator |
| `Property_Area` | Area classification of the property |

### Target variable

| Target | Description |
|---|---|
| `Loan_Status` | Loan approval outcome: `Y` for approved and `N` for rejected |

The target variable is encoded numerically during preprocessing:

- `Y` → `1` (Approved)
- `N` → `0` (Rejected)

The `Loan_ID` column is removed before model training because it serves as an identifier rather than an applicant characteristic.

**Note:** The original dataset file is not included in this repository unless it has been uploaded separately. Place the CSV file in the project directory before running the notebook.

## 🛠️ Technologies and Libraries

### Programming language
- Python

### Libraries

| Library | Purpose |
|---|---|
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization and correlation heatmaps |
| Scikit-learn | Model training, data splitting, and evaluation |
| Jupyter Notebook | Interactive development and documentation |

## ⚙️ Project Workflow

### 1. Data Loading and Initial Inspection

The dataset is loaded into a Pandas DataFrame.

Initial inspection is performed to understand the structure of the dataset, examine the available columns, review descriptive statistics, and identify the data types of individual features.

The following operations are used:

- `head()` to inspect the first few rows.
- `describe()` to examine descriptive statistics.
- `dtypes` to inspect column data types.
- `isnull().sum()` to identify missing values.
- `duplicated().sum()` to check for duplicate records.

### 2. Data Preprocessing

Data preprocessing prepares the raw dataset for machine learning.

The following steps are performed:

**Handling missing values**

- Categorical columns such as `Married`, `Dependents`, and `Self_Employed` are imputed using their mode.
- Numerical columns such as `LoanAmount` and `Loan_Amount_Term` are imputed using their median.
- `Credit_History` is imputed using its median.
- Missing values in `Gender` are handled using the mode after numerical encoding.

**Encoding categorical variables**

Categorical features are mapped to numerical values so they can be used by the classifier.

Examples include:

- Gender: Male = 1, Female = 0
- Married: Yes = 1, No = 0
- Education: Graduate = 1, Not Graduate = 0
- Self-employed: Yes = 1, No = 0
- Property Area: Rural = 0, Semiurban = 1, Urban = 2

The `Dependents` value `3+` is converted to `3` and the column is converted to an integer type.

**Removing unnecessary columns**

The `Loan_ID` column is removed from the modeling dataset, and the processed data is stored in `df_clean`.

### 3. Exploratory Data Analysis (EDA)

Exploratory data analysis is used to investigate patterns in the data before model training.

The notebook includes:

- Descriptive statistical analysis.
- Correlation matrix and heatmap.
- Applicant income distribution by loan status.
- Loan amount distribution by loan status.
- Loan approval rate by credit history.
- Loan approval rate by property area.

These visualizations help explore the relationships between applicant characteristics and loan approval outcomes.

### 4. Train-Test Split

The dataset is divided into input features (`X`) and the target variable (`y`).

The following configuration is used:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Enabled using the target variable

Stratified splitting helps preserve the class proportions of the target variable across the training and testing sets.

### 5. Model Building

A Decision Tree Classifier from Scikit-learn is used to train the model.

The classifier learns decision rules from the training data to distinguish between approved and rejected loan applications.

The following hyperparameters are used:

| Parameter | Value | Purpose |
|---|---|---|
| `max_depth` | 3 | Limits the maximum depth of the tree |
| `min_samples_leaf` | 5 | Requires at least five training samples in each leaf |
| `random_state` | 42 | Supports reproducibility |

Limiting the tree depth and setting the minimum leaf size helps control model complexity and reduce the risk of overfitting.

### 6. Model Evaluation

The trained classifier predicts loan outcomes for the test dataset.

The notebook evaluates the predictions using:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many predicted positive cases are actually positive.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1-score:** Combines precision and recall into a single metric.
- **Classification report:** Summarizes precision, recall, F1-score, and support for each class.
- **Confusion matrix:** Provides a breakdown of correct and incorrect classifications.

These metrics help evaluate model performance beyond accuracy alone.

The actual evaluation results should be recorded after executing the notebook.

### 7. Decision Tree Visualization

The trained decision tree is visualized using Scikit-learn's `plot_tree()` function.

The visualization displays the learned decision rules, feature-based splits, and predicted classes at the tree's nodes.

This helps explain how the classifier makes predictions and makes the model's decision process easier to inspect.

### 8. Feature Importance Analysis

Feature importance values are extracted from the trained classifier using `feature_importances_`.

These values indicate the relative contribution of each input feature to the tree's splitting decisions.

The results are stored in a DataFrame containing the feature names and their corresponding importance values.

Feature importance provides a useful way to explore which variables the model relies on most. It does not establish that a feature causes loan approval.

### 9. Sample Applicant Prediction

The notebook also demonstrates how to make a prediction for a sample applicant using the trained classifier.

The model returns:

- Predicted class: Approved or Rejected.
- Predicted class probabilities for approval and rejection.

The sample applicant's features must follow the same order and encoding used during training.

The displayed probability is the model's estimated class probability, not a guarantee of the real-world outcome.

## 🧠 Algorithm: Decision Tree Classifier

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks.

It works by repeatedly splitting the dataset according to feature-based decision rules. Each split aims to separate the observations into more homogeneous groups.

The tree consists of:

- **Root node:** The first decision point.
- **Internal nodes:** Intermediate decisions based on features.
- **Branches:** Outcomes of the decision rules.
- **Leaf nodes:** Final predicted classes.

Decision trees are relatively easy to interpret and can capture nonlinear relationships between input features and the target.

In this project, the classifier learns decision rules from historical loan application data to predict loan approval status.

## 📂 Repository Structure

The repository can be organized as follows:

```text
loan-approval-prediction-decision-tree/
│
├── Decision_tree_loan_approval.ipynb
├── train_u6lujuX_CVtuZ9i (1).csv
├── README.md
```

The CSV file is optional in the repository structure if you prefer to keep the dataset separate. If it is not included, users must obtain the dataset independently and update the notebook's file path.

## 🚀 Installation and Setup

### Prerequisites

Install Python and Jupyter Notebook, or use Google Colab to run the notebook online.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-prediction-decision-tree.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Navigate to the project directory

```bash
cd loan-approval-prediction-decision-tree
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

Place the loan prediction CSV file in the project directory.

If you are running the notebook locally, update the data-loading statement to:

```python
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")
```

The uploaded notebook currently uses a Google Colab path, so update that path if you run the project outside Colab.

### Step 5: Run the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Decision_tree_loan_approval.ipynb` and execute the cells in sequence.

Alternatively, upload the notebook and dataset to Google Colab and run them there.

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 📈 Results

The notebook generates the following outputs:

- Dataset inspection and preprocessing results.
- Correlation heatmap and exploratory visualizations.
- Training and testing dataset sizes.
- Test accuracy and classification report.
- Trained decision tree visualization.
- Feature importance table.
- Sample applicant predictions and predicted class probabilities.

**Model performance:** The test accuracy and other evaluation metrics should be filled in after running the notebook. Results are intentionally not specified here because they have not been verified.

## 🌟 Key Learnings

Through this project, I explored and practiced:

- Data cleaning and missing-value imputation.
- Categorical feature encoding.
- Exploratory data analysis and correlation analysis.
- Training and testing data separation.
- Supervised machine learning with a Decision Tree Classifier.
- Model evaluation using classification metrics.
- Decision tree visualization and interpretability.
- Feature importance analysis.
- Generating predictions for new input records.

## 🔮 Future Improvements

Potential improvements to this project include:

- Compare Decision Tree performance with Logistic Regression, Random Forest, and other classification algorithms.
- Tune hyperparameters using cross-validation and grid search.
- Evaluate class imbalance and use suitable evaluation metrics.
- Build a reusable preprocessing and model-training pipeline.
- Add feature importance visualizations.
- Create an interactive interface using Streamlit.
- Save the trained model and support predictions through a simple application.
- Evaluate fairness, data leakage risks, and the limitations of historical lending data before any real-world use.

## ⚠️ Limitations and Responsible Use

This project is intended for educational and experimental purposes.

Predictions depend on the quality and representativeness of the dataset. Historical patterns may not reflect current lending policies, and the model may produce incorrect predictions.

The model should not be used as the sole basis for real lending decisions. Any practical deployment would require additional validation, fairness assessments, privacy safeguards, and appropriate human oversight.

## 👨‍💻 Author

**Your Name**

- GitHub: [YOUR-GITHUB-USERNAME](https://github.com/Vinaytr2005)
- LinkedIn: [[Add your LinkedIn profile URL](https://www.linkedin.com/in/vinay-tr-computer-datascience-engineering/)]

## 📄 License

This project can be distributed under the MIT License if you choose to include one. Add a `LICENSE` file to the repository before claiming that the project is licensed under MIT.

---

⭐ If you find this project useful, feel free to explore the code, learn from the workflow, and suggest improvements.
