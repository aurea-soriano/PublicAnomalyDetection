from kenchi.outlier_detection.statistical import HBOS

hbos = HBOS(novelty=True).fit(X)
y_pred = hbos.predict(X)
