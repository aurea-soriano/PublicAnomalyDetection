from sklearn.cluster import AgglomerativeClustering

clusters = 3
y_pred = AgglomerativeClustering(n_clusters=clusters).fit_predict(X)


from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

clusters=5
cls = linkage(X, method='ward')
y_pred = fcluster(cls, t=clusters, criterion='maxclust')

dendrogram(cls)
plt.show()
