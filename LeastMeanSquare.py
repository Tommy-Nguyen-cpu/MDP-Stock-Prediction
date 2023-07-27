class LeastMeanSquare:
    # Given a list of historical stock prices and the times, find weights that can find the change in data.
    def CalculateWeights(w0, w1, w2, alpha, stockprices, times, dailyChanges, iterations):
        for index in range(iterations):
            for dataIndex in range(len(dailyChanges)):
                yhat = LeastMeanSquare.CalculateYHat(w0, w1, w2, stockprices[dataIndex], times[dataIndex])
                (w0,w1, w2) = LeastMeanSquare.NewWeights(w0, w1, w2, yhat, dailyChanges[dataIndex], alpha, stockprices[dataIndex], times[dataIndex])
        
        return w0, w1, w2

    def CalculateYHat(w0, w1, w2, price, time):
        return w0 + w1 * price + w2 * time

    def NewWeights(w0, w1, w2, yhat, y, alpha, price, time):
        newW0 = LeastMeanSquare.Calculate(w0, alpha, yhat, y, 1)
        newW1 = LeastMeanSquare.Calculate(w1, alpha, yhat, y, price)
        newW2 = LeastMeanSquare.Calculate(w2, alpha, yhat, y, time)
        return newW0, newW1, newW2

    def Calculate(w, alpha, yhat, y, param):
        return w - alpha * (yhat - y) * param