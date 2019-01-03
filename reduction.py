# -*- coding: utf-8 -*
import numpy as np
from sklearn import decomposition, manifold
import pickle
import itertools
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import operator
from functools import reduce


# from feature_count import get_input_feature_from_one_section, get_feature_word, FeatureCount
file_read = open('input_features.bin', 'rb')
s = file_read.read()
input_features = pickle.loads(s)
file_read.close()

high_dim_input = []
for section_id in input_features:
    high_dim_input.append(input_features[section_id])

high_dim_input = np.array(list(itertools.chain.from_iterable(high_dim_input)))
labels = []
for section_id in input_features:
    for i in range(len(input_features[section_id])):
        labels.append(section_id)
labels = np.array(labels)

markers1 = ['0', '1', '2', '3']
markers2 = ['4', '5', '6', '7']
markers3 = ['8', '9', '10', '11']
markers4 = ['12', '13', '14', '15', '16', '17']
markers5 = ['18', '19']
markers6 = ['20']
markers7 = ['21']
markers8 = ['22']

'''PCA降维'''
# PCA 2维
pca = decomposition.PCA(n_components=2)
X_input = pca.fit_transform(high_dim_input)

print('Percentage of variance explained by each of the selected components:', pca.explained_variance_ratio_)

ax2 = plt.figure().add_subplot(111)

for i, marker in zip([0, 1, 2, 3], markers1):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='r', s=10, marker='^'.format(marker))

for i, marker in zip([4, 5, 6, 7], markers2):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='#FFC0CB', s=10, marker='o'.format(marker))

for i, marker in zip([8, 9, 10, 11], markers3):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='g', s=10, marker='s'.format(marker))

for i, marker in zip([12, 13, 14, 15, 16, 17], markers4):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='b', s=10, marker='x'.format(marker))

for i, marker in zip([18, 19], markers5):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='b', s=10, marker='x'.format(marker))

for i, marker in zip([20], markers6):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='k', s=10, marker='p'.format(marker))


for i, marker in zip([21], markers7):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='c', s=10, marker='p'.format(marker))

for i, marker in zip([22], markers8):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color='m', s=10, marker='p'.format(marker))


'''
'''


# PCA 3维
pca = decomposition.PCA(n_components=3)
X_input = pca.fit_transform(high_dim_input)
print('Percentage of variance explained by each of the selected components:', pca.explained_variance_ratio_)


# colors = ['#48A946', '#E55523', '#E5E223', '#23E5DF', '#F70DB4', '#0D77F7', '#CD2E7C', '#F70D80']


s = []
ax3 = plt.figure().add_subplot(111, projection='3d')
plt.xlim([-150, 150])
plt.ylim([-150, 150])

for i, marker in zip([0, 1, 2, 3], markers1):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color='r', s=10, marker='^'.format(marker))

for i, marker in zip([4, 5, 6, 7], markers2):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color='#FFC0CB', s=10, marker='o'.format(marker))


for i, marker in zip([8, 9, 10, 11], markers3):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color='g', s=10, marker='s'.format(marker))

for i, marker in zip([12, 13, 14, 15, 16, 17], markers4):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color='b', s=10, marker='x'.format(marker))


for i, marker in zip([18, 19], markers5):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color='b', s=10, marker='x'.format(marker))

plt.show()
plt.close()

'''
plt.legend((s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]),
        ('1_1.txt', '1_2.txt', '1_3.txt', '1_4.txt',
         '2_1.txt', '2_2.txt', '2_3.txt', '2_4.txt'), loc='lower left')
'''

# ax.title('1234 vs 5678')




'''
# TSNE降维

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
X_tsne = tsne.fit_transform(high_dim_input)
# plot_embedding_2d(X_tsne[:, 0:2], 't-SNE 2D')

colors1 = [
        'y', 'y',
        'y', 'y',
        ]
colors2 = [
        'b', 'b',
        'b', 'b']


# colors = ['#48A946', '#E55523', '#E5E223', '#23E5DF', '#F70DB4', '#0D77F7', '#CD2E7C', '#F70D80']

markers1 = ['1', '2', '3', '4']
markers2 = ['5', '6', '7', '8']
s = []
# 2维
ax2 = plt.figure().add_subplot(111)
for color, i, marker in zip(colors1, [0, 1, 2, 3], markers1):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color=color, s=20, marker='^'.format(marker))
for color, i, marker in zip(colors2, [4, 5, 6, 7], markers2):
    ax2.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color=color, s=20, marker='o'.format(marker))

# 3维
ax3 = plt.figure().add_subplot(111, projection='3d')
for color, i, marker in zip(colors1, [0, 1, 2, 3], markers1):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color=color, s=10, marker='^'.format(marker))
for color, i, marker in zip(colors2, [4, 5, 6, 7], markers2):
    ax3.scatter(X_input[labels == i, 0], X_input[labels == i, 1], X_input[labels == i, 2],
                 color=color, s=10, marker='o'.format(marker))



plt.show()
'''