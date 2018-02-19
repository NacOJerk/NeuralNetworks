import numpy as np

#Some pretty shitty code warning

def sigmoid(x):
  return 1 / (1. + np.exp(-x))

def dev_sigmoid(x):
    res = sigmoid(x)
    return res - np.power(res, 2)

if __name__ == "__main__":
    #np.random.seed(0)
    weights = []
    bias = []
    io = []
    inp = int(input("how many nodes do you want: "))
    prevLayer = inp
    while True:
        amount = input("Enter amount of neuron per layer (type stop to stop): ")
        if amount == "stop":
            weights.append(2 * np.matrix(np.random.rand(prevLayer, 1) - 1, dtype=np.float64))
            bias.append(np.matrix(np.random.rand(1, 1), dtype=np.float64))
            break
        amount = int(amount)
        weights.append(2 * np.matrix(np.random.rand(prevLayer, amount) - 1, dtype=np.float64))
        bias.append(np.matrix(np.random.rand(1, amount), dtype=np.float64))
        prevLayer = amount
    eWeights = []
    eBias = []
    for w in weights:
        eWeights.append(np.dot(0., w))
    for b in bias:
        eBias.append(np.dot(0., b))
    while True:
        inp = input("Please enter input (type 'stop' to stop): ")
        if inp == "stop":
            break
        nums = inp.split(" ")
        nums = [nums[:-1], nums[-1]]
        nums[1] = round(float(nums[-1]), 2)
        for j in range(len(nums[0])):
            nums[0][j] = round(float(nums[0][j]), 2)
        print("Adding {}".format(nums))
        io.append(nums)
    generation = 0
    while True:
        avgW = eWeights
        avgB = eBias
        test = 0
        for i in range(len(io)):
            aPrev = (io[i][0])
            netList = []
            for j in range(len(weights)):
                z = np.dot(aPrev, weights[j]) + bias[j]
                a = sigmoid(z)
                netList.append((aPrev, z))
                if generation % 10000 == 0:
                    #print("L: {} a(L-1): {} z(L): {} a(L): {}".format(j + 1, aPrev, z, a))
                    pass
                aPrev = a
            cost = (a - io[i][1])**2
            if generation % 10000 == 0:
                #print("Weights: ")
                #print(weights)
                #print("Bias: ")
                #print(bias)
                print("Generation: {} input: {} ouput: {} desire: {} error: {}".format(generation, io[i][0], a.A[0][0], io[i][1], cost))
            if round(cost.A[0][0], 5) == 0:
                test += 1
            costDer = 2 * (a - io[i][1])
            for j in range(len(netList)):
                spot = len(netList) - j - 1
                zDer = np.matrix(dev_sigmoid(netList[spot][1]).A * costDer.A, dtype=np.float64)
                avgW[spot] += (zDer.T * netList[spot][0]).reshape(avgW[spot].shape)
                avgB[spot] += zDer
                costDer = (zDer * weights[spot].T)
        avgW = np.divide(avgW, len(io))
        for j in range(len(avgB)):
            avgB[j] = np.divide(avgB[j], len(io))
        for j in range(len(avgW)):
            weights[j] = weights[j] - avgW[j]
        for j in range(len(avgB)):
            bias[j] = bias[j] - avgB[j]
        if test == len(io):
            break
        generation +=1
    print("Finished generation: {}".format(generation))
    while True:
        inp = input("Test Input (stop to stop): ")
        if inp == "stop":
            break
        nums = inp.split(" ")
        for j in range(len(nums)):
            nums[j] = round(float(nums[j]), 2)
        aPrev = nums
        for j in range(len(weights)):
            z = np.dot(aPrev, weights[j]) + bias[j]
            a = sigmoid(z)
            netList.append((aPrev, z))
            aPrev = a
        print(round(a.A[0][0], 2))
