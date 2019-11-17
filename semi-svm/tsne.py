import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read in the input
print "Reading the CSV file"
def load_data():
    train_data = np.genfromtxt('mental-state.csv', dtype=np.float32, delimiter=',')
    X_raw = train_data[:, :-1].astype(np.float32)
    y = train_data[:, -1].astype(np.int32)
    return X_raw, y

[X_raw,y] = load_data()
print(X_raw, y)
print(("Number of samples before removing class 1: " + str(X_raw.shape[0])))

# Remove one class label to make it binary
X_raw = np.delete(X_raw, np.where((y==1)), axis=0)
y = np.delete(y, np.where((y==1)), axis=0)/2

print(X_raw, y)
print(("Number of samples after removing class 1: " + str(X_raw.shape[0])))

# Remove the infinity values
X_raw[np.where(np.isinf(X_raw))] = 0

# Run TSNE
X_embedded = TSNE(n_components=2, perplexity=50, n_iter=2000).fit_transform(X_raw)

# Plot TSNE
fig = plt.figure()

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='winter')
plt.title(('TSNE plot of the data'))
plt.savefig('tsne.png')
