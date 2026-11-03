# Spam Email Classification with Naive Bayes

Machine learning project for detecting spam emails from their textual content. The project implements an end-to-end classification pipeline, from data preparation and exploratory analysis to text vectorization, model training, hyperparameter tuning and evaluation.

## Objective

The main objective of this project is to develop and evaluate machine learning models for classifying emails as **spam** or **ham** based on their textual content.

The project covers the complete machine learning workflow:

* Data preparation and preprocessing
* Exploratory data analysis
* Text vectorization
* Model training
* Hyperparameter tuning
* Cross-validation
* Evaluation on a held-out test set
* Comparison of different classification models

## Dataset

The dataset used in this project was constructed by combining two publicly available datasets hosted on Kaggle:

* [Spam email Dataset]{https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset}— by _w1998. The dataset originally contained **5,728 emails**.
* [Spam-Ham Mail Classification Dataset]{https://www.kaggle.com/datasets/yuvanshudadhich/spam-ham-dataset} — by Yuvanshu Dadhich and Chetan Pemmasani. The dataset originally contained **9,354 emails**.

After loading both datasets, they were harmonized and combined into a single dataset containing **15,082 emails**. Before cleaning, there were **3,210 duplicate entries** and no missing values.

After removing duplicate entries, the final dataset contained **11,872 unique emails**, with no missing values.

The target variable is `spam`:

| Label     | Class |     Emails | Percentage |
| --------- | ----- | ---------: | ---------: |
| 0         | Ham   |     8,648 |     72.84% |
| 1         | Spam  |      3,224 |     27.16% |
| **Total** |       | **11,872** |   **100%** |

### Data quality

Before preprocessing, the dataset contained:

* **3,210 duplicate entries**
* No missing values in the `text` or `spam` columns

The duplicate removal step was applied before training the models to avoid having repeated emails in the dataset.

## Data Preparation & Preprocessing

The two source datasets contain email information in different formats. Their fields were therefore harmonized into a common structure containing:

* `text`: email textual content, which included the header and the body of the emails
* `spam`: binary target variable

The datasets were then combined, duplicate entries were removed and missing values were checked.

### Text preprocessing

Two approaches to text preprocessing were evaluated:

1. Retaining the original email text, except for the english stopwords, and a few HTML and URL tags too persistent in both groups
2. Removing HTML tags, URLs and other formatting-related elements.

The cleaning strategy was evaluated experimentally. In this dataset, removing these elements resulted in slightly lower classification performance across the tested models.

Therefore, the final models retain the **original email text**. The HTML/URL cleaning functionality remains available as an optional preprocessing step.

## Exploratory Data Analysis

The exploratory analysis includes:

* Spam/ham class distribution
* Missing-value analysis
* Duplicate detection
* Most frequent words in the dataset
* Inspection of common email and HTML-related terms

These analyses were used to understand the dataset before applying machine learning models.

## Text Representation

Machine learning models require numerical representations of the email text. Two vectorization approaches were evaluated:

### CountVectorizer

`CountVectorizer` represents each email using the frequency of the words appearing in the document.

Both unigrams and bigrams were considered.

### TF-IDF

`TfidfVectorizer` was also evaluated to represent words according to their importance within the dataset.

Both unigrams and bigrams were considered.

The vectorizers are included directly within the Scikit-learn pipelines so that they are fitted only on the corresponding training data during cross-validation.

## Models

Three classification algorithms were evaluated:

### Multinomial Naive Bayes

Multinomial Naive Bayes is particularly suited to text classification problems represented using word counts or TF-IDF features.

### Support Vector Classifier

A Support Vector Classifier was evaluated as a second approach for separating spam and ham emails in the resulting feature space.

### Random Forest

Random Forest was included as a tree-based classification approach to provide a different modelling strategy for comparison.

## Hyperparameter Tuning

Hyperparameter optimization was performed using `GridSearchCV` with **5-fold cross-validation**.

The search included different combinations of:

* Count-based and TF-IDF representations
* Unigrams and bigrams
* Model-specific hyperparameters
* Class weighting where applicable

The models were selected based on their cross-validated **F1-score**.

## Evaluation

After hyperparameter tuning, the selected models were evaluated on a separate held-out test set.

The following metrics were used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **ROC-AUC**

Additional evaluation was performed using:

* ROC curves
* Confusion matrices

Particular attention was given to the precision, recall and F1-score of the **spam class**, since correctly identifying spam is the main objective of the classification task.

## Results

The final models are compared using their performance on the held-out test set and their cross-validation F1-score.

| Model                   | Precision | Recall |     F1 | Accuracy |  CV F1 | AUC |
| ----------------------- | --------: | -----: | -----: | -------: | -----: | --: |
| Multinomial Naive Bayes |    0.9921 | 0.9752 | 0.9836 |   0.9912 | 0.9752 |  1.00 |
| SVC                     |    0.9795 | 0.9628 | 0.9711 |   0.9844 | 0.9688 |  1.00 |
| Random Forest           |    0.9692 | 0.9752 | 0.9722 |   0.9848 | 0.9545 |  1.00 |

The results show high classification performance across all three evaluated models. The preprocessing experiment also showed that retaining the original email text provided slightly better performance than removing HTML and URL information.

## Project Structure

```text
Spam_Classification-Naive_Bayes/
├── data/
│   └── raw/
│       ├── emails.csv
│       └── spam_ham_dataset.csv
├── models/
│   └── best_models.joblib
├── notebooks/
│   └── main.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── results_test.joblib
│   ├── roc_curves.png
│   └── confusion_matrices.png
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.py
├── environment.yml
├── README.md
└── .gitignore
```

### Main components

* `preprocessing.py` — loads and prepares the datasets and creates the train/test split.
* `training.py` — defines the machine learning pipelines and performs hyperparameter tuning.
* `evaluation.py` — evaluates the trained models and generates comparison metrics and plots.
* `main.py` — orchestrates the complete pipeline.
* `main.ipynb` — documents the analysis, visualizations and results.
* `models/` — stores the trained models.
* `results/` — stores evaluation results and generated plots.

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/Robert-Cuello/Spam_Classification-Naive_Bayes.git
cd Spam_Classification-Naive_Bayes
```

### 2. Download the datasets

Download the datasets from Kaggle, and place them in `data/raw`:
* `emails.csv`
* `spam_ham_dataset,csv`

### 3. Create the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate spam_classifier
```

### 4. Run the project

The complete pipeline can be executed with:

```bash
python src/main.py
```

The execution options can be controlled in `main.py`:

```python
TRAIN = False
EVALUATE = True
PLOT = True
```

* `TRAIN = True` trains the models using `GridSearchCV`.
* `EVALUATE = True` evaluates the trained models on the test set.
* `PLOT = True` generates the ROC curves and confusion matrices.

The trained models can be loaded from `models/best_models.joblib` to avoid repeating the computationally expensive training process.

## Limitations

Some limitations of the current project include:

* The dataset is composed of a limited collection of email sources and may not represent all types of modern spam.
* The models are trained using the textual content of the available emails and may not generalize equally well to other email distributions.
* The project does not currently evaluate temporal changes in spam patterns or concept drift.
* The vocabulary generated by the text vectorizers can become very large, increasing model size and computational requirements.

## Future Improvements

Possible extensions of the project include:

* Evaluation on an independent external dataset
* More advanced NLP representations
* Model deployment
* Containerization
* Development of an API for making predictions

## Author

**Robert Cuello**

[GitHub](https://github.com/Robert-Cuello)
