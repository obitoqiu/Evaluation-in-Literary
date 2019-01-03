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
from pylab import *

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
print('labels:', labels)

markers1 = ['0', '1', '2', '3']
markers2 = ['4', '5', '6', '7']
markers3 = ['8', '9', '10', '11']
markers4 = ['12', '13', '14', '15', '16', '17']
markers5 = ['18', '19']
markers6 = ['20']
markers7 = ['21']
markers8 = ['22']

colors = ['r', 'r', '#FFC0CB', 'g', 'b', 'k', 'c', 'm']
markers = ['^', 'o', 's', 'x', 'p']
mk=''
k=1
'''PCA降维'''
pca = decomposition.PCA(n_components=8)
X_input = pca.fit_transform(high_dim_input)
sum = 0
for one_dim in pca.explained_variance_ratio_:
    sum = sum + float(one_dim)

print('Percentage of variance explained total selected components:', sum)


# kmeans聚类 效果略差于birch
def kmeans(X_input, k):
    from sklearn.cluster import KMeans
    print('start k-means cluster:')
    clusterer = KMeans(n_clusters=k, init='k-means++')  # 设置聚类模型
    y = clusterer.fit_predict(X_input)
    print(y)
    print('点到簇的质心距离和：', clusterer.inertia_)
    return y,clusterer.inertia_


# birch聚类
def birch(X_input, k):
    from sklearn.cluster import Birch
    print('start birch cluster:')
    clusterer = Birch(n_clusters=k,threshold=1)
    y = clusterer.fit_predict(X_input)
    print(y)
    return y


# AP聚类
def dbscan(X_input):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    print('start dbscan cluster:')
    clusterer = DBSCAN(eps=79,min_samples=30)
    y = clusterer.fit_predict(X_input)


    print(y)

    return y



# 计算轮廓系数
def Silhouette(X_input, y):
    from sklearn.metrics import silhouette_samples, silhouette_score
    print('计算轮廓系数([-1,1]越大越好)：')
    silhouette_avg = silhouette_score(X_input, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X_input, y)  # 每个点的轮廓系数
    print(silhouette_avg)
    return silhouette_avg, sample_silhouette_values


# 画轮廓系数分析图
def DrawSi(silhouette_avg, sample_silhouette_values, y, k):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # 创建一个 subplot with 1-row 2-column
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.3, 1.0])

    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    ax1.set_ylim([0, len(y) + (k + 1) * 10])

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i]

        #fill_between(定义曲线的x坐标, 曲线1的y坐标, 曲线2的y坐标, 点色, 边色, 透明度)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)
        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper + 10

     # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")

    plt.show()

# 画聚类后的图
def Drawclu2D(pre_y, high_dim_input):

    ax2 = plt.figure().add_subplot(111)
    pca = decomposition.PCA(n_components=2)

    # print(len(pre_y),len(high_dim_input))

    X = pca.fit_transform(high_dim_input)
    for i in range(len(X)):

        x = X[i,0]
        y = X[i,1]
        color_id = pre_y[i]

        ax2.scatter(x, y, color=colors[color_id], s=10, marker='^')
    plt.show()
    plt.close()


def find_k(X_input):

    iner = []
    for i in range(1,7):
        y,d= kmeans(X_input,i)
        iner.append(d)
    plot(range(1, 7), iner)
    show()

if __name__ == "__main__":
    # y = birch(X_input, k)
    find_k(X_input)
    y,_=kmeans(X_input,k)
    Drawclu2D(y, high_dim_input)
    silhouette_avg, sample_silhouette_values = Silhouette(X_input, y) # 轮廓系数
    DrawSi(silhouette_avg, sample_silhouette_values, y, k)

