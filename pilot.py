
from io import StringIO
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

from time import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

X_path = sys.argv[1]
y_path = sys.argv[2]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



print('Splitting Data and Labels')

X = pd.read_csv(X_path, header= None, names=['index', 'data'], error_bad_lines=False)
X = X.data

Y = pd.read_csv(y_path, header= None, names=['index', 'label'], error_bad_lines=False)
Y = Y.label

class_names = list(set(Y))

print('Splitting Training and Testing Data')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

print('Extracting Features')
vect = TfidfVectorizer(ngram_range = (1, 1), max_features = 25000)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

print(X_test_dtm.shape)

print('Training Model')
t0 = time()
mlpClassifier = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=200)
mlpClassifier.fit(X_train_dtm, Y_train)
train_time = time() - t0
print("train time: %0.3fs" % train_time)


print('Predicting & Calculating Model Accuracy')
t0 = time()
y_pred_class = mlpClassifier.predict(X_test_dtm)
test_time = time() - t0
print("test time:  %0.3fs" % test_time)


print('Accuracy & confusion matrix')

print('Accuracy : ')
print(metrics.accuracy_score(y_test, y_pred_class))
print('F1 Score : ')
print(metrics.f1_score(y_test, y_pred_class, average="weighted"))
print('Precision_score :')
print(metrics.precision_score(y_test, y_pred_class, average="weighted"))
print('Recall_score :')
print(metrics.recall_score(y_test, y_pred_class, average="weighted") )
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_class)
print(cnf_matrix)
print(classification_report(y_test,y_pred_class))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
