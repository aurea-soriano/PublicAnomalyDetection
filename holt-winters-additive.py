from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit = ExponentialSmoothing(data, seasonal_periods=periodicity, trend='add', seasonal='add').fit(use_boxcox=True)
fit.fittedvalues.plot(color='blue')
fit.forecast(5).plot(color='green')
plt.show()
