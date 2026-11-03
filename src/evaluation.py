import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Obtaining predictions and classification reports for the best models on the test set
def evaluate_models(best_models, X_test, y_test):
    results_test = {}

    for name, data in best_models.items():
        best_estimator = data['estimator']
        y_pred = best_estimator.predict(X_test)
        
        results_test[name] = classification_report(
            y_test, 
            y_pred, 
            output_dict=True, 
            target_names=['ham', 'spam'])

        if hasattr(best_estimator, "predict_proba"):
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_estimator.decision_function(X_test)
        
        results_test[name]['y_pred'] = y_pred
        results_test[name]['y_proba'] = y_proba

    joblib.dump(results_test, './results/results_test.joblib')

    ## Comparing different models on the test split
    roc_data = {}

    models = [
        'multinomial_nb',
        'svc',
        'random_forest'
    ]

    for model in models:
        y_proba = results_test[model]['y_proba']
        fpr, tpr, thresholds = roc_curve(y_test, results_test[model]['y_proba'])
        auc = roc_auc_score(y_test, y_proba)

        roc_data[model] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        } 

    df_comparison = pd.DataFrame({
        name: {
            'precision_spam': metrics['spam']['precision'],
            'recall_spam': metrics['spam']['recall'],
            'f1_spam': metrics['spam']['f1-score'],
            'accuracy': metrics['accuracy'],
            'cv_f1_train': best_models[name]['cv_score']
        }

    for name, metrics in results_test.items()}).T

    df_comparison['auc'] = df_comparison.index.map(lambda x: roc_data[x]['auc'])

    df_comparison.sort_values(by='f1_spam', ascending=False)
    df_comparison.to_csv('./results/model_comparison.csv', index=True)

    return results_test, roc_data, df_comparison


def plot_roc_auc(roc_data):
    plt.figure(figsize=(8, 6), dpi=100)

    for i, model, data in roc_data.items():
        plt.plot(roc_data[model]['fpr'], roc_data[model]['tpr'], label = f"{model} (AUC = {roc_data[model]['auc']:.2f})")

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for Classifiers')
    plt.grid(True)
    plt.legend()
    plt.savefig('./results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results_test, y_test):
    models = list(results_test.keys())
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, model in enumerate(models):
        y_pred = results_test[model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, ax = axes_flat[i], annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Not Spam'], yticklabels=['Spam', 'Not Spam'])
        axes_flat[i].set_title(f'Confusion Matrix for {model}')
        axes_flat[i].set_xlabel('Predicted')
        axes_flat[i].set_ylabel('Actual')

    plt.suptitle('Confusion Matrix for the tested models')
    plt.tight_layout() 
    plt.savefig('./results/confusion_matrices.png', dpi=300, bbox_inches='tight') 
    plt.show()