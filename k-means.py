from sklearn.cluster import KMeans

clusters = 3
y_pred = KMeans(n_clusters=clusters).fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=y_pred)
plt.show()
