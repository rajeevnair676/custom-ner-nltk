import pickle
import warnings
warnings.filterwarnings('ignore')

from itertools import chain
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import confusion_matrix,classification_report

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    Function to print a custom confusion matrix since CRFsuite doesn't provide one. Function reference in the references section
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def get_confusion_matrix(y_true,y_pred,labels):
    """
    Function to build a confusion matrix since CRFsuite doesn't have one
    """
    trues,preds = [], []
    for yseq_true, yseq_pred in zip(y_true, y_pred):
        trues.extend(yseq_true)
        preds.extend(yseq_pred)
    print_cm(confusion_matrix(trues,preds,labels=labels),labels)


def train_seq(X_train,Y_train,X_dev,Y_dev,model_name):
    """
    Function to train and save the CRF model, and print the evaluation metrics
    """
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50)
    crf.fit(X_train, Y_train)
    labels = list(crf.classes_)
    pickle.dump(crf, open(model_name, 'wb'))
    #testing:
    y_pred = crf.predict(X_dev)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_f1_score(Y_dev, y_pred,average='weighted', labels=labels))
    print(classification_report(list(chain.from_iterable(Y_dev)),list(chain.from_iterable(y_pred)), labels=sorted_labels))
    get_confusion_matrix(Y_dev, y_pred,labels=sorted_labels)