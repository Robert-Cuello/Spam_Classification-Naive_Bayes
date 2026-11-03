if __name__ == "__main__":
    import joblib
    from src.preprocessing import preparing_data
    from src.training import train_models
    from src.evaluation import evaluate_models, plot_roc_auc, plot_confusion_matrices

    TRAIN = False
    EVALUATE = True
    PLOT = True

    df, personalised_stop_words, X_train, X_test, y_train, y_test = preparing_data(clean_html_urls=False)

    if TRAIN:
        best_models = train_models(
            X_train,
            y_train,
            personalised_stop_words
        )
    else:
        best_models = joblib.load("./models/best_models.joblib")

    if EVALUATE:
        results_test, roc_data, df_comparison = evaluate_models(
            best_models,
            X_test,
            y_test
        )

        if PLOT:
            plot_roc_auc(roc_data)
            plot_confusion_matrices(results_test, y_test)