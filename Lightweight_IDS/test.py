import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def test_sklearn_installation():
    # Mostrar la versión instalada
    print("scikit-learn version:", sklearn.__version__)

    # Cargar dataset de ejemplo
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Separar en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Entrenar un clasificador sencillo
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    # Evaluar en el set de prueba
    score = clf.score(X_test, y_test)
    print(f"Test accuracy: {score:.2f}")

if __name__ == "__main__":
    test_sklearn_installation()
