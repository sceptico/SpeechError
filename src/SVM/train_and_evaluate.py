import prepare_data
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump

def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.1):
    """Evaluate model with an adjustable decision threshold."""
    y_scores = model.decision_function(X_eval)
    y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

    accuracy = accuracy_score(y_eval, y_eval_pred)
    precision = precision_score(y_eval, y_eval_pred, average='binary')
    recall = recall_score(y_eval, y_eval_pred, average='binary')
    f1 = f1_score(y_eval, y_eval_pred, average='binary')
    auc = roc_auc_score(y_eval, y_scores)
    conf_matrix = confusion_matrix(y_eval, y_eval_pred)

    print(classification_report(y_eval, y_eval_pred))
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return accuracy, precision, recall, f1, auc, conf_matrix

def main():
    # Calculate target_length from training data
    target_length = prepare_data.get_max_sequence_length("data/metadata/label_train_resampled.csv", "data/contextual_features")
    print(f"Using target length of {target_length} based on maximum sequence length in training data.")

    # Load training data
    X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
        target_length=target_length,
        primary_feature_dir="data/contextual_features",
        secondary_feature_dir="data/resampled_features"
    )

    # Load evaluation data
    X_eval, y_eval = prepare_data.load_eval_data(
        target_length=target_length,
        primary_feature_dir="data/contextual_features",
        scaler=scaler
    )

    # Baseline SVM model with linear kernel
    print("\nTraining baseline model with linear kernel (C=1.0)")
    baseline_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
    baseline_model.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Evaluate baseline model on training data
    print("\nTraining Data Results for the baseline model")
    baseline_threshold = 0.1  # Set the baseline threshold if needed
    evaluate_model_with_threshold(baseline_model, X_train, y_train, threshold=baseline_threshold)

    # Evaluate baseline model on evaluation data with custom threshold
    print("\nEvaluation Results for the baseline model with linear kernel")
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=baseline_threshold)

    # Define the parameter grid to include both linear and RBF kernels
    param_grid = [
        {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
        # {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    ]

    # Set up StratifiedKFold for cross-validation to maintain class distribution
    skf = StratifiedKFold(n_splits=5)

    # Set up GridSearchCV with F1 scoring, focusing on the minority class
    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring='f1', cv=skf, n_jobs=-1, verbose=2)

    # Train with grid search
    grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best model parameters: {best_params}")

    # Evaluate on the evaluation set with a custom threshold
    print("\nEvaluation Results for the best model with custom threshold")
    custom_threshold = 0.1  # Set your desired threshold
    accuracy, precision, recall, f1, auc, conf_matrix = evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=custom_threshold)

    # Evaluate the best model on training data
    print("\nTraining Data Results for the best model")
    evaluate_model_with_threshold(best_model, X_train, y_train, threshold=custom_threshold)

    # Save the best model
    dump(best_model, 'best_svm_model.joblib')
    print(f"\nBest model saved as best_svm_model.joblib with F1 Score: {f1} and AUC-ROC: {auc}")

if __name__ == "__main__":
    main()



# import prepare_data
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from joblib import dump

# def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.1):
#     """Evaluate model with an adjustable decision threshold."""
#     y_scores = model.decision_function(X_eval)
#     y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

#     accuracy = accuracy_score(y_eval, y_eval_pred)
#     precision = precision_score(y_eval, y_eval_pred, average='binary')
#     recall = recall_score(y_eval, y_eval_pred, average='binary')
#     f1 = f1_score(y_eval, y_eval_pred, average='binary')
#     auc = roc_auc_score(y_eval, y_scores)
#     conf_matrix = confusion_matrix(y_eval, y_eval_pred)

#     print(classification_report(y_eval, y_eval_pred))
#     print(f"Threshold: {threshold}")
#     print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     return accuracy, precision, recall, f1, auc, conf_matrix

# def main():
#     # Calculate target_length from training data
#     target_length = prepare_data.get_max_sequence_length("data/metadata/label_train_resampled.csv", "data/contextual_features")
#     print(f"Using target length of {target_length} based on maximum sequence length in training data.")

#     # Load training data
#     X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features",
#         secondary_feature_dir="data/resampled_features"
#     )

#     # Load evaluation data
#     X_eval, y_eval = prepare_data.load_eval_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features",
#         scaler=scaler
#     )

#     # Baseline SVM model with linear kernel
#     print("\nTraining baseline model with linear kernel (C=1.0)")
#     baseline_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
#     baseline_model.fit(X_train, y_train, sample_weight=sample_weights_train)

#     # Evaluate baseline model with custom threshold
#     print("\nEvaluation Results for the baseline model with linear kernel")
#     baseline_threshold = 0.0  # Set the baseline threshold if needed
#     evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=baseline_threshold)

#     # Define the parameter grid to include both linear and RBF kernels
#     param_grid = [
#         {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
#         {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
#     ]

#     # Set up StratifiedKFold for cross-validation to maintain class distribution
#     skf = StratifiedKFold(n_splits=5)

#     # Set up GridSearchCV with F1 scoring, focusing on the minority class
#     grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring='f1', cv=skf, n_jobs=-1, verbose=2)

#     # Train with grid search
#     grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)

#     # Get the best model from grid search
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_
#     print(f"Best model parameters: {best_params}")

#     # Evaluate on the evaluation set with a custom threshold
#     print("\nEvaluation Results for the best model with custom threshold")
#     custom_threshold = 0.1  # Set your desired threshold
#     accuracy, precision, recall, f1, auc, conf_matrix = evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=custom_threshold)

#     # Evaluate the best model on training data
#     print("\nTraining Data Results for the best model")
#     evaluate_model_with_threshold(best_model, X_train, y_train, threshold=custom_threshold)

#     # Save the best model
#     dump(best_model, 'best_svm_model.joblib')
#     print(f"\nBest model saved as best_svm_model.joblib with F1 Score: {f1} and AUC-ROC: {auc}")

# if __name__ == "__main__":
#     main()




# import prepare_data
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from joblib import dump

# def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.1):
#     """Evaluate model with an adjustable decision threshold."""
#     y_scores = model.decision_function(X_eval)
#     y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

#     accuracy = accuracy_score(y_eval, y_eval_pred)
#     precision = precision_score(y_eval, y_eval_pred, average='binary')
#     recall = recall_score(y_eval, y_eval_pred, average='binary')
#     f1 = f1_score(y_eval, y_eval_pred, average='binary')

#     print(classification_report(y_eval, y_eval_pred))
#     print(f"Threshold: {threshold}")
#     print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
#     return accuracy, precision, recall, f1

# def main():
#     # Calculate target_length from training data
#     target_length = prepare_data.get_max_sequence_length("data/metadata/label_train_resampled.csv", "data/contextual_features")
#     print(f"Using target length of {target_length} based on maximum sequence length in training data.")

#     # Load training data
#     X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features",
#         secondary_feature_dir="data/resampled_features"
#     )

#     # Load evaluation data
#     X_eval, y_eval = prepare_data.load_eval_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features",
#         scaler=scaler
#     )

#     # Baseline SVM model with linear kernel
#     print("\nTraining baseline model with linear kernel (C=1.0)")
#     baseline_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
#     baseline_model.fit(X_train, y_train, sample_weight=sample_weights_train)

#     # Evaluate baseline model with custom threshold
#     print("\nEvaluation Results for the baseline model with linear kernel")
#     baseline_threshold = 0.0  # Set the baseline threshold if needed
#     evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=baseline_threshold)

#     # Define the parameter grid for RBF kernel
#     param_grid = {
#         'C': [1, 10, 100],
#         'gamma': [0.001, 0.01]
#     }

#     # Initialize SVC with RBF kernel and balanced class weight
#     # svc = SVC(kernel='rbf', class_weight='balanced')


#     # Initialize SVC with linear kernel and balanced class weight
#     svc = SVC(kernel='linear', class_weight='balanced')

#     # Define the parameter grid for the linear kernel (gamma is not applicable to linear kernel)
#     param_grid = {
#         'C': [0.0001, 0.001, 0.01,0.1, 1, 10, 100]  # Range of values for C
#     }

#     # Set up GridSearchCV with F1 scoring
#     grid_search = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)

#     # Train with grid search
#     grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)

#     # Get the best model from grid search
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_
#     print(f"Best model parameters: {best_params}")

#     # Evaluate on the evaluation set with a custom threshold
#     print("\nEvaluation Results for the best model with custom threshold")
#     custom_threshold = 0.1  # Set your desired threshold
#     accuracy, precision, recall, f1 = evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=custom_threshold)

#     # Save the best model
#     dump(best_model, 'best_svm_model.joblib')
#     print(f"\nBest model saved as best_svm_model.joblib with F1 Score: {f1}")

# if __name__ == "__main__":
#     main()


