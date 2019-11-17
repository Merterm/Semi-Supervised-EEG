import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#   Read in the input
print("Reading the CSV file")
def load_data():
    train_data = np.genfromtxt('mental-state.csv', dtype=np.float32, delimiter=',')
    X_raw = train_data[:, :-1].astype(np.float32)
    y = train_data[:, -1].astype(np.int32)
    return X_raw, y

[X_raw,y] = load_data()
print(X_raw, y)
print(("Number of samples before removing class 1: " + str(X_raw.shape[0])))

#Remove one class label to make it binary
X_raw = np.delete(X_raw, np.where((y==1)), axis=0)
y = np.delete(y, np.where((y==1)), axis=0)/2

print(X_raw, y)
print(("Number of samples after removing class 1: " + str(X_raw.shape[0])))

#Remove the infinity values
X_raw[np.where(np.isinf(X_raw))] = 0

'''
def load_data():
   data_train = np.genfromtxt('mental.csv',
                               dtype=np.float64, delimiter=',',
                               usecols=np.arange(0,2549))
   #data_test = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',')
   #'test_x': data_test[:, :-1].astype(np.float64),
   #'test_y': data_test[:, -1].astype(np.int64),

   return {
       'all': data_train,
       'train_x': data_train[:, :-1],
       'train_y': data_train.transpose()[-1],
   }

import numpy as np
from scipy import stats


mental = load_data()
train_all_data = mental['all']
train_x_data = mental['train_x']
#print(len(train_x_data[0] ))
train_x_data_copy = np.copy(train_x_data)
train_x_data_copy = train_x_data_copy.transpose()
train_y_data = mental['train_y']
#print(train_y_data,train_x_data_copy[1])
benchmark = 0.3
corrlations = stats.pearsonr(train_x_data_copy[0], train_y_data)[0]
idx = np.array([])
for i in range(len(train_x_data_copy)):
   this_col = stats.pearsonr(train_x_data_copy[i], train_y_data)[0]
   if this_col >= benchmark:
       idx = np.append(idx, [i])
idx = np.append(idx, [-1]).astype(int)
#print(idx)

x = train_all_data[:,idx]
np.savetxt('mental_clean.csv', x, delimiter=',', fmt='%s')
'''

# Do PCA
#pca = PCA(n_components=30)
#pca.fit(X_raw)
#X = pca.transform(X_raw)
X = X_raw

#   Split the data into test and train data
test_pcnt = 0.15
X_train = X[:int(len(X)*(1-test_pcnt)),:]
X_test = X[int(len(X)*(1-test_pcnt)):,:]
y_train = y[:int(len(X)*(1-test_pcnt))]
y_test = y[int(len(X)*(1-test_pcnt)):]

print(X_test)

#Gamma Series
gammas = np.array([10**(i) for i in range (0,-20,-1)])
#SVM Using rbf Kernel Function, test on different gamma values
train_error =  np.array([])
test_error = np.array([])

for gamma in gammas:
    mental_svm = SVC(gamma=300, kernel = 'rbf')
    mental_svm.fit(X_train, y_train)

    train_result = mental_svm.predict(X_train)
    print(train_result)
    train_false = (train_result !=  y_train)
    train_error_rate = np.sum(train_false)/len(train_false)
    train_error = np.append(train_error, [train_error_rate])
    print("Train error:", train_error_rate)

    test_result = mental_svm.predict(X_test)
    print(test_result)
    test_correct = (test_result ==  y_test)
    test_error_rate = float(len(test_correct[test_correct == False]))/len(test_correct)
    print(gamma)
    test_error = np.append(test_error, [test_error_rate])
