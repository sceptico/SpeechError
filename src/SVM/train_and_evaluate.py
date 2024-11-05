import prepare_data
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump

def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.0):
    """Evaluate model with an adjustable decision threshold."""
    y_scores = model.decision_function(X_eval)
    y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

    accuracy = accuracy_score(y_eval, y_eval_pred)
    precision = precision_score(y_eval, y_eval_pred, average='binary')
    recall = recall_score(y_eval, y_eval_pred, average='binary')
    f1 = f1_score(y_eval, y_eval_pred, average='binary')

    print(classification_report(y_eval, y_eval_pred))
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

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

    # Evaluate baseline model with custom threshold
    print("\nEvaluation Results for the baseline model with linear kernel")
    baseline_threshold = 0.0  # Set the baseline threshold if needed
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=baseline_threshold)

    # Define the parameter grid for RBF kernel
    param_grid = {
        'C': [1, 10, 100],
        'gamma': [0.001, 0.01]
    }

    # Initialize SVC with RBF kernel and balanced class weight
    # svc = SVC(kernel='rbf', class_weight='balanced')


    # Initialize SVC with linear kernel and balanced class weight
    svc = SVC(kernel='linear', class_weight='balanced')

    # Define the parameter grid for the linear kernel (gamma is not applicable to linear kernel)
    param_grid = {
        'C': [0.0001, 0.001, 0.01,0.1, 1, 10, 100]  # Range of values for C
    }

    # Set up GridSearchCV with F1 scoring
    grid_search = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)


    # # Set up GridSearchCV with F1 scoring
    # grid_search = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)

    # Train with grid search
    grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best model parameters: {best_params}")

    # Evaluate on the evaluation set with a custom threshold
    print("\nEvaluation Results for the best model with custom threshold")
    custom_threshold = 0.2  # Set your desired threshold
    accuracy, precision, recall, f1 = evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=custom_threshold)

    # Save the best model
    dump(best_model, 'best_svm_model.joblib')
    print(f"\nBest model saved as best_svm_model.joblib with F1 Score: {f1}")

if __name__ == "__main__":
    main()








'''
before apply a baseline model, we need to prepare the data linear kernel and C=1.0
'''
# import prepare_data
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from joblib import dump

# def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.0):
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

#     # Define the parameter grid for RBF kernel
#     param_grid = {
#         'C': [1, 10, 100],
#         'gamma': [0.001, 0.01]
#     }

#     # Initialize SVC with RBF kernel and balanced class weight
#     svc = SVC(kernel='rbf', class_weight='balanced')

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
#     custom_threshold = 0.2  # Set your desired threshold
#     accuracy, precision, recall, f1 = evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=custom_threshold)

#     # Save the best model
#     dump(best_model, 'best_svm_model.joblib')
#     print(f"\nBest model saved as best_svm_model.joblib with F1 Score: {f1}")

# if __name__ == "__main__":
#     main()



'''
code below is used to train and evaluate the SVM model before scaling up to the full dataset
also, without tuning hyperparameters
'''

# import prepare_data
# import svm
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from joblib import dump

# def evaluate_model(model, X_eval, y_eval):
#     """Evaluate a given model on the evaluation set and print metrics."""
#     y_eval_pred = model.predict(X_eval)
#     accuracy = accuracy_score(y_eval, y_eval_pred)
#     precision = precision_score(y_eval, y_eval_pred, average='binary')
#     recall = recall_score(y_eval, y_eval_pred, average='binary')
#     f1 = f1_score(y_eval, y_eval_pred, average='binary')
#     print(classification_report(y_eval, y_eval_pred))
#     print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
#     return accuracy, precision, recall, f1

# def main():
#     # Calculate target_length once from training data, using primary contextual feature directory
#     target_length = prepare_data.get_max_sequence_length("data/metadata/label_train_resampled.csv", "data/contextual_features")
#     print(f"Using target length of {target_length} based on maximum sequence length in training data.")

#     # Load training data from both directories (contextual and resampled) with sample weights
#     X_train, y_train, sample_weights_train = prepare_data.load_train_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features",
#         secondary_feature_dir="data/resampled_features"
#     )

#     # Load evaluation data only from the contextual features directory
#     X_eval, y_eval = prepare_data.load_eval_data(
#         target_length=target_length,
#         primary_feature_dir="data/contextual_features"
#     )

#     # Define multiple model configurations for comparison
#     model_configs = [
#         {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
#         {"kernel": "rbf", "C": 0.5, "gamma": "scale"},
#         {"kernel": "linear", "C": 1.0},
#         {"kernel": "poly", "C": 1.0, "degree": 3},
#     ]

#     best_model = None
#     best_f1 = 0
#     best_config = None

#     # Iterate over each model configuration
#     for config in model_configs:
#         print(f"\nTraining model with config: {config}")
#         model = svm.create_svm_model(**config)
        
#         # Train the model with sample weights
#         model.fit(X_train, y_train, sample_weight=sample_weights_train)

#         # Evaluate on the evaluation set
#         print(f"\nEvaluation Results for config: {config}")
#         accuracy, precision, recall, f1 = evaluate_model(model, X_eval, y_eval)

#         # Keep track of the best model based on F1 score
#         if f1 > best_f1:
#             best_f1 = f1
#             best_model = model
#             best_config = config

#     # Save the best model for testing
#     if best_model:
#         dump(best_model, 'best_svm_model.joblib')
#         print(f"\nBest model saved as best_svm_model.joblib with config: {best_config}")
#         print(f"Best F1 Score on Evaluation Set: {best_f1}")

# if __name__ == "__main__":
#     main()
