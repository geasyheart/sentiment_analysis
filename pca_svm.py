import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
import pandas as pd
from sklearn import metrics

df = pd.read_csv("data/2000_data.csv")
y = df.iloc[:, 1]
x = df.iloc[:, 2:]

# PCA降维
# #计算全部贡献率
n_components = 400
pca = PCA(n_components=n_components)
pca.fit(x)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_, len(pca.explained_variance_))
# #PCA作图
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_)  # , linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')import matplotlib.pyplot as plt
# from sklearn import svm
plt.show()

x_pca = PCA(n_components=100).fit_transform(x)

# Create ROC curve

clf = svm.SVC(C=2, probability=True)
clf.fit(x_pca, y)
print("Test Accuracy: ", clf.score(x_pca, y))

pred_probas = clf.predict_proba(x_pca)[:, 1]  # score

fpr, tpr, _ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()
