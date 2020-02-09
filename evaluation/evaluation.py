from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def get_evaluation_metrics(y_test, model_pred):
    """Inset y_test and model_predictions. Print (1) confusion matrix, (2) classification report and (3) final accuracy score"""
    print("Get classification metrics")
    classes = np.unique(model_pred)
    print("\nClassification Report \n", classification_report(y_test, model_pred))
    print("\nConfusion Matrix \n", confusion_matrix(y_test, model_pred, labels=classes))
    print("\nAccuracy Score \n", accuracy_score(y_test, model_pred))


def dl_accuracy_score(dl_model, X_test, y_test):
    """Input deep learning model, test features and y features. Print accuracy score of the model."""
    scores = dl_model.evaluate(X_test, y_test, batch_size=64)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def plot(model, history):
    print("Plot model history")
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def evaluation_main(y_test, X_test, model, model_name, history=None):
    models = ["lstm", "bilstm", "gru"]
    if model_name in models:
        print(model_name)
        # dl_accuracy_score(model, X_test, y_test)
        y_test = np.argmax(y_test, axis=1)
        X_test = np.argmax(X_test, axis=1)
    
        model_pred = model.predict_classes(X_test)
        get_evaluation_metrics(y_test, model_pred)
        plot(model, history)
        
    else:
        model_pred = model.predict(X_test)
        get_evaluation_metrics(y_test, model_pred)

    print("Finished Evaluation")

if __name__ == "__main__":
    evaluation_main()



