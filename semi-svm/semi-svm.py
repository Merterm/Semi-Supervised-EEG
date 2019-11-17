import numpy as np
import random
import matplotlib.pyplot as plt
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from sklearn.decomposition import PCA
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel
from examples.plotutils import evaluate_and_plot

#   Read in the input
print "Reading the CSV file"
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

comps = range(1,21)#[2, 10, 20, 30, 100, 200, 500, 800, 1000, X_raw.shape[0], X_raw.shape[1]]
semi_accs = np.zeros(len(comps))
sup_accs = np.zeros(len(comps))
cnt = 0
num_trials = 50

print X_raw.shape

for j in comps:
    print j
    if j != X_raw.shape[1]:
        # Do PCA
        pca = PCA(n_components=j)
        pca.fit(X_raw)
        X = pca.transform(X_raw)
    else:
        X = X_raw

    #   Split the data into test and train data
    test_pcnt = 0.15
    X_train = X[:int(len(X)*(1-test_pcnt)),:]
    X_test = X[int(len(X)*(1-test_pcnt)):,:]
    y_train = y[:int(len(X)*(1-test_pcnt))]
    y_test = y[int(len(X)*(1-test_pcnt)):]

    ytrue = y_train

    print X_train.shape, y_train.shape


    # Just supervised score
    basemodel = WQDA() # weighted Quadratic Discriminant Analysis
    #basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
    basemodel.fit(X_train, ytrue)
    print "full labeled wqda score", basemodel.score(X_test, y_test)
    print "standard error of wqda", 1.96 * np.sqrt(basemodel.score(X_test, y_test)*(1-basemodel.score(X_test, y_test))/X_test.shape[0])

    # Just supervised score
    #basemodel = WQDA() # weighted Quadratic Discriminant Analysis
    basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
    basemodel.fit(X_train, ytrue)
    print "full labeled log.reg. score", basemodel.score(X_test, y_test)
    print "standard error of log reg", 1.96 * np.sqrt(basemodel.score(X_test, y_test)*(1-basemodel.score(X_test, y_test))/X_test.shape[0])

    super_acc = np.zeros(num_trials)
    semi_acc = np.zeros(num_trials)
    sum_super = 0
    sum_super_err = 0
    sum_semi_err = 0
    sum_semi = 0
    for i in range(num_trials):
        print ".",
        # label a few points
        labeled_N = 10
        ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
        random_labeled_points = random.sample(np.where(ytrue == 0)[0], labeled_N/2)+\
                                random.sample(np.where(ytrue == 1)[0], labeled_N/2)
        ys[random_labeled_points] = ytrue[random_labeled_points]

        #print ys

        X_model = X_train

        # supervised score
        basemodel = WQDA() # weighted Quadratic Discriminant Analysis
        #basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
        basemodel.fit(X_model[random_labeled_points, :], ys[random_labeled_points])
        #print "supervised log.reg. score", basemodel.score(X_test, y_test)

        #if j == 2:
            #Plot the base model
        #    evaluate_and_plot(basemodel, X_model, ys, ytrue, "Logistic Regression", subplot = 1, block=True)

        #Calculate accuracy
        sum_super += basemodel.score(X_test, y_test)
        super_acc[i] = basemodel.score(X_test, y_test)
        sum_super_err += 1.96 * np.sqrt(super_acc[i]*(1-super_acc[i])/X_test.shape[0])

        # fast (but naive, unsafe) self learning framework
        ssmodel = SelfLearningModel(basemodel)
        ssmodel.fit(X_model, ys)
        #print "self-learning log.reg. score", ssmodel.score(X_test, y_test)

        #if j == 2:
            #Plot the ssmodel
        #    evaluate_and_plot(ssmodel, X_model, ys, ytrue, "Self-Learning", subplot = 2, block=True)

        #Calculate accuracy
        sum_semi += ssmodel.score(X_test, y_test)
        semi_acc[i] = ssmodel.score(X_test, y_test)
        sum_semi_err += 1.96 * np.sqrt(semi_acc[i]*(1-semi_acc[i])/X_test.shape[0])

        #if j==2:
            #Save the figure
        #    plt.savefig(('comparisons_' + str(j) + '_' + str(i) + '.png'))

    print "average supervised accuracy: ", sum_super/num_trials
    sup_accs[cnt] = sum_super/num_trials
    #print "standard deviation of supervised accuracy: ", np.std(super_acc, ddof=1)
    print "standard error of supervised: ", sum_super_err/num_trials

    print "average semi-supervised accuracy: ", sum_semi/num_trials
    semi_accs[cnt] = sum_semi/num_trials
    #print "standard deviation of semi-supervised accuracy: ", np.std(semi_acc, ddof=1)
    print "standard error of semi-supervised: ", sum_semi_err/num_trials

    cnt = cnt+1

print sup_accs
print semi_accs

plt.plot(comps, sup_accs)
plt.plot(comps, semi_accs)
plt.legend(['Average Supervised Accuracy', 'Average Semi-Supervised Accuracy'])
plt.xlabel('Number of Features')
plt.ylabel('Average Accuracy')
plt.title(('Change in Average Accuracy of ' + str(num_trials) + ' Trials using PCA'))
plt.savefig('pca.png')
