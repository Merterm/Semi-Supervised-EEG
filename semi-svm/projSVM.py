def load_data():
    data_train = np.genfromtxt('mental_clean.csv', 
                                dtype=np.float64, delimiter=',',
                                usecols=np.arange(0,48)) #48-mental #785-test
    #data_test = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',')
    #'test_x': data_test[:, :-1].astype(np.float64),
    #'test_y': data_test[:, -1].astype(np.int64),

    return {
        'all': data_train,
        'train_x': data_train[:, :-1],
        'train_y': data_train.transpose()[-1],
    }
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import xlwt


mental = load_data()
x = mental['train_x']
train_len = int(len(x)/2)
train_x = x[:train_len, :]
test_x = x[train_len:, :]
y = mental['train_y']
train_y = y[:train_len]
test_y = y[train_len:]
#assert(x.shape==(2360,47))
#assert(y.shape==(2360,))


#mental_svm = LinearSVC(penalty = 'l2', C = 1)

#scores = mental_svm.decision_function(test_x) 
#print(scores[:5])
#print(result[:100])



#Gamma Series 
gammas = np.array([10**(i) for i in range (0,-20,-1)])
#SVM Using rbf Kernel Function, test on different gamma values
train_error =  np.array([])
test_error = np.array([])
for gamma in gammas:
	mental_svm = SVC(gamma = gamma, kernel = 'rbf')
	mental_svm.fit(train_x, train_y)

	train_result = mental_svm.predict(train_x)
	train_correct = (train_result ==  train_y)
	train_error_rate = float(len(train_correct[train_correct == False]))/len(train_correct)
	train_error = np.append(train_error, [train_error_rate])

	test_result = mental_svm.predict(test_x)
	test_correct = (test_result ==  test_y)
	test_error_rate = float(len(test_correct[test_correct == False]))/len(test_correct)
	#print(gamma)
	test_error = np.append(test_error, [test_error_rate])

#Import to Excel 
train_error = [train_error]
test_error = [test_error]
gammas = [gammas]
SVM_result = np.append(gammas, train_error, axis = 0)
SVM_result = np.append(SVM_result, test_error, axis = 0)
np.savetxt('SVM_result.csv', SVM_result, delimiter=',', fmt='%s')



Cs = np.array([1])
Cs = np.append(Cs, [10*(i+1) for i in range (100)])
#SVM Using rbf Kernel Function, fix gamma to be 10^(-7) and
#test on different penalties
train_error =  np.array([])
test_error = np.array([])
for c in Cs:
	mental_svm = SVC(gamma = 10**(-7), C = c, kernel = 'rbf')
	mental_svm.fit(train_x, train_y)

	train_result = mental_svm.predict(train_x)
	train_correct = (train_result ==  train_y)
	train_error_rate = float(len(train_correct[train_correct == False]))/len(train_correct)
	train_error = np.append(train_error, [train_error_rate])

	test_result = mental_svm.predict(test_x)
	test_correct = (test_result ==  test_y)
	test_error_rate = float(len(test_correct[test_correct == False]))/len(test_correct)
	#print(gamma)
	test_error = np.append(test_error, [test_error_rate])

#Import to Excel 
train_error = [train_error]
test_error = [test_error]
Cs = [Cs]
nuSVM_result = np.append(Cs, train_error, axis = 0)
nuSVM_result = np.append(nuSVM_result, test_error, axis = 0)
np.savetxt('nuSVM_result.csv', nuSVM_result, delimiter=',', fmt='%s')







