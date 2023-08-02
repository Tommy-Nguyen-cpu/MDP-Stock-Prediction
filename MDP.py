import LeastMeanSquare as LMS
import ActionEnum as actions
import copy
class MDP:
    def __init__(self, stocks, weights, times, low, medium, high):
        print("Got to MDP.")
        self.U = {}
        self.R = {}
        self.T = {}
        self.Policy = {}
        self.Weight = .5
        self.PotentialActions = [actions.ActionEnum.SELL, actions.ActionEnum.HOLD, actions.ActionEnum.BUY]
        self.UpVals = []
        self.DownVals = []
        self.StableVals = []
    
    def InitializeMDP(self, stockDatas, weights, times, low, medium, high):
        print("Initializing...")

        # Initialize the utility function to be the initial reward function.
        currentPrice = stockDatas[0]
        for time in times:
            newPrice = LMS.LeastMeanSquare.CalculateYHat(weights[0], weights[1], weights[2], currentPrice, time)
            self.R[time] = newPrice - currentPrice
            self.U[time] = self.R[time]

            self.Policy[time] = []
            self.Policy[time] += self.PotentialActions

            name = self.GetStateName(self.U[time])
            # print("State: " + name)
            if name == "Up":
                self.UpVals.append(self.U[time])
            elif name == "Down":
                self.DownVals.append(self.U[time])
            elif name == "Stable":
                self.StableVals.append(self.U[time])
            # print("New cost: " + str(self.R[time]))
        
        # Set up transition table.
        states = ["Up", "Stable", "Down"]
        for state in states:
            for action in self.PotentialActions:
                self.T[(state, action)] = {}
                for state2 in states:
                    self.T[(state, action)].update({state2: 0})
        
        changesInPrices = []
        for current, future in zip(stockDatas[1]["Close"], stockDatas[1]["Close"][1:]):
            change = future - current
            if change < -1:
                changesInPrices.append("Down")
            elif change > -1 and change < 1:
                changesInPrices.append("Stable")
            elif change > 1:
                changesInPrices.append("Up")
        
        stateTransition = []
        for change1, change2 in zip(changesInPrices, changesInPrices[1:]):
            action = None
            if change2 == "Up":
                action = actions.ActionEnum.BUY
            elif change2 == "Stable":
                action = actions.ActionEnum.HOLD
            elif change2 == "Down":
                action = actions.ActionEnum.SELL
            stateTransition.append([change1, change2, action])
            # print(change1 + " to " + change2 + " with action " + str(action))
        
        for fromState in self.T.keys():
            for toState in self.T[fromState].keys():
                fromStateActionToStateCounter = 0
                fromStateActionCounter = 0
                for pair in stateTransition:
                    if fromState[0] == pair[0] and toState == pair[1] and fromState[1] == pair[2]:
                        fromStateActionToStateCounter += 1
                        fromStateActionCounter += 1
                    elif fromState[0] == pair[0] and toState != pair[1]:
                        fromStateActionCounter += 1
                if fromStateActionCounter > 0:
                    self.T[fromState][toState] = fromStateActionToStateCounter / fromStateActionCounter
                else:
                    self.T[fromState][toState] = 0
                # print(fromState[0] + " to " + toState + " probability: " + str(self.T[fromState][toState]))
        
        # print("state: " + str(self.U))
        self.ValueIteration()
        # print("new state: " + str(self.U))

        bestActions = []
        for key in self.U.keys():
            bestActions.append(self.MaxAction(key, self.PotentialActions)[1])
        # print("Optimal Policy: " + str(bestActions))
        return self.ResultActionProbability(bestActions)
    
    def ResultActionProbability(self, optimalPolicy):
        probBuy = optimalPolicy.count(actions.ActionEnum.BUY) / len(optimalPolicy)
        probSell = optimalPolicy.count(actions.ActionEnum.SELL) / len(optimalPolicy)
        probHold = optimalPolicy.count(actions.ActionEnum.HOLD) / len(optimalPolicy)
        return probBuy, probSell, probHold
    
    def GetProbability(self, state, action, nextState):
        current = self.GetStateName(state)
        futureState = self.GetStateName(nextState)

        return self.T[(current, action)][futureState]

    def GetStateName(self, state):
        if state < -1:
            return "Down"
        elif state > -1 and state < 1:
            return "Stable"
        elif state > 1:
            return "Up"

    def MaxAction(self, stateKey, actions):

        meanUpVal = 0
        if len(self.UpVals) > 0:
            meanUpVal = sum(self.UpVals) / len(self.UpVals)
        
        meanDownVal = 0
        if len(self.DownVals) > 0:
            meanDownVal = sum(self.DownVals) / len(self.DownVals)
        
        meanStableVal = 0
        if len(self.StableVals) > 0:
            meanStableVal = sum(self.StableVals) / len(self.StableVals)


        policies = []
        for action in actions:
            upProb = self.GetProbability(self.U[stateKey], action, meanUpVal)
            downProb = self.GetProbability(self.U[stateKey], action, meanDownVal)
            stableProb = self.GetProbability(self.U[stateKey], action, meanStableVal)
            policySum = upProb * meanUpVal + downProb * meanDownVal + stableProb * meanStableVal
            policies.append(policySum)

        maximumPolicy = max(policies)
        indexOfMax = policies.index(maximumPolicy)

        return maximumPolicy, actions[indexOfMax]

    def ValueIteration(self):
        for state in self.U.keys():
            self.U[state] = self.R[state]
        changes = 100
        while changes > 1:
            UPrime = copy.deepcopy(self.U)
            for state in self.U.keys():
                UPrime[state] = self.R[state] + self.Weight * self.MaxAction(state, self.Policy[state])[0]
                
                # Minimum changes are kept track of.
                absoluteChange = abs(UPrime[state])
                if absoluteChange < changes:
                    changes = changes - absoluteChange
            self.U = UPrime
