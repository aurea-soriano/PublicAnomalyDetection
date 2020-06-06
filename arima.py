from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

p = 5  # lag
d = 1  # difference order
q = 0  # size of moving average window

train, test = train_test_split(X, test_size=0.20, shuffle=False)
history = train.tolist()
predictions = []

for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	fit = model.fit(disp=False)
	pred = fit.forecast()[0]

	predictions.append(pred)
	history.append(test[t])

print('MSE: %.3f' % mean_squared_error(test, predictions))

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
