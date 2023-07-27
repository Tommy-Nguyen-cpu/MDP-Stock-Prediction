import yfinance as yf
import MDP as myMDP
import LeastMeanSquare as LMS
import csv

# Print out to test our equal frequency binning algorithm.
def WriteToCSV(excelDocName, headerRow, subsequentRows):
    filename = excelDocName + ".csv"

    with open(filename, mode = 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(headerRow)
        for subRow in subsequentRows:
            writer.writerow(subRow)


# Converts our data point into digestable strings ("H", "M", or "L" for high, medium, or low respectively)
def EqualFrequencyBinning(data):
    binSize = 3
    print("Binning...")

    # Sorts data into ascending order.
    sortedData = sorted(data, key= lambda obj : obj)

    # Separates data into "low", "medium", and "high" bins.
    low = []
    low += sortedData[0: int(len(sortedData)/binSize)]
    medium = []
    medium += sortedData[int(len(sortedData)/binSize):int((len(sortedData)/binSize)*2)]
    high = []
    high = sortedData[int((len(sortedData)/binSize)*2):int((len(sortedData)/binSize)*3)]

    # Gives the data points a "label" depending on which bin the data point was found in.
    dataLabels = []
    for dataPoint in data:
        if dataPoint in low:
            dataLabels.append("L")
        elif dataPoint in medium:
            dataLabels.append("M")
        elif dataPoint in high:
            dataLabels.append("H")
    
    # WriteToCSV(data, sortedData, dataLabels)
    return low, medium, high

def LabelData(history):
    # TODO: We're concerned with the difference between the previous day closed.
    dailyChanges = []
    for index in range(len(history["Close"])):
        dailyChanges.append(history["Close"][index] - history["Open"][index])
    return EqualFrequencyBinning(dailyChanges)
    # closedDataLabels = EqualFrequencyBinning(history["Close"])
    # openDataLabels = EqualFrequencyBinning(history["Open"])

def LinearRegression(history):
    changes = []
    times = []
    for index in range(1, len(history["Close"])):
        changes.append(history["Close"][index])
        times.append(index)
    
    w0 = -100
    w1 = -100
    w2 = -100
    (w0, w1, w2) = LMS.LeastMeanSquare.CalculateWeights(w0, w1, w2, .00001, history["Close"], times, changes, 200000)
    print("Final weights: " + str(w0) + ", " + str(w1) + "," + str(w2))
    return w0, w1, w2, times

def main():
    
    companies = [ "TSLA", "WBA", "AMZN"]
    HistoricalStockData = {}

    for company in companies:
        stock = yf.Ticker(company)

        # Retrieves the current price of the stock.
        currentPrice = stock.info["currentPrice"]

        # Retrieves the history of the stock for the past 42 days.
        hist = stock.history(period="42d")
        

        HistoricalStockData[company] = (currentPrice, hist)

    weights = []

    headerRow = ["Stock", "Current Price", "Probability to Buy", "Probability to Hold", "Probability to Sell"]
    stockActions = []
    for key, data in HistoricalStockData.items():
        print("Current Cost: " + str(data[0]) + " for " + key)

        (w0,w1,w2, times) = LinearRegression(data[1])

        weights = []
        weights += [w0,w1,w2]
        (lowBin, mediumBin, highBin) = LabelData(data[1])
        mdp = myMDP.MDP(data, weights, times, lowBin, mediumBin, highBin)
        (probBuy, probSell, probHold) = mdp.InitializeMDP(data, weights, times, lowBin, mediumBin, highBin)
        probBuy = round(probBuy, 2) * 100
        probSell = round(probSell,2) * 100
        probHold = round(probHold, 2) * 100
        stockActions.append([key, str(data[0]), str(probBuy), str(probHold), str(probSell)])
        print("Prob Buy: " + str(probBuy) + ", prob Sell: " + str(probSell) + ", prob Hold: " + str(probHold))
    WriteToCSV("StockPredictionResult", headerRow, stockActions)

if __name__ == "__main__":
    main()