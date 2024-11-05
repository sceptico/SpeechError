from sklearn.svm import SVC

def create_svm_model(kernel="rbf", C=1.0, gamma="scale", degree=3, class_weight="balanced"):
    """
    Creates and returns an SVM model with specified configuration, including degree for poly kernel if applicable,
    and class weight handling for imbalanced classes.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight=class_weight)
    print(f"SVM model created with kernel={kernel}, C={C}, gamma={gamma}, degree={degree if kernel == 'poly' else 'N/A'}, class_weight={class_weight}")
    return model

if __name__ == "__main__":
    model = create_svm_model()
    print("SVM model created with kernel:", model.kernel)
