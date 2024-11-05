# test_SVM.py

import prepare_data
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from joblib import load

def main():
    # Load the test data
    X_test, y_test = prepare_data.load_test_data()

    # Load the trained model
    model = load('best_svm_model.joblib')
    print("Loaded model from best_svm_model.joblib")

    # Evaluate on the test set
    print("\nFinal Evaluation Results on Test Set:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))
    print("Precision on Test Set:", precision_score(y_test, y_test_pred, average='binary'))
    print("Recall on Test Set:", recall_score(y_test, y_test_pred, average='binary'))
    print("F1 Score on Test Set:", f1_score(y_test, y_test_pred, average='binary'))

if __name__ == "__main__":
    main()
