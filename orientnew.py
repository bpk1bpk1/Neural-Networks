import numpy
import math
hiddenlist = {}
outputlist = {}
alpha = 0.001

def neural_nets(testfile,best):
    input = []
    with open(testfile) as f:
        for line in f:
            countlist = 0
            angle = 0
            input = []
            for word in line.split():
                if(countlist == 1):
                    angle = word
                if(countlist >= 2):
                    input.append(word)
                countlist += 1
            hidden(input,best,angle)

def hidden(input,best,angle):
    hiddennode = []
    value = 0
    for i in range(0,best):
        sum = 0
        for j in range(0,len(input)):
            value = float(hiddenlist[j][i]) * float(input[j])
            sum = sum + value
        hiddennode.append(sum)
    count = len(input)
    input_outputlayer = activation_function(hiddennode)
    output(input_outputlayer,angle,hiddennode,count,best)

def output(input,angle,hiddennode,count,best):
    outputnode = []
    for i in range(0,4):
        sum = 0
        for j in range(0,len(input)):
            value = float(outputlist[j][i]) * float(input[j])
            sum = sum + value
        outputnode.append(sum)
    value = softmax_function(outputnode)
    backpropogation(value,angle,outputnode,hiddennode,count,best)

def backpropogation(value,angle,outputnode,hiddennode,inputcount,hiddencount):
    global hiddenlist
    global outputlist
    groundtruth = [0,0,0,0]
    deltaj = []
    if(angle == 0):
        groundtruth[0] += 1
    elif(angle == 90):
        groundtruth[1] += 1
    elif(angle == 180):
        groundtruth[2] += 1
    else:
        groundtruth[3] += 1
    for j in range(len(value)):
        r = derivative(value[j])
        valuej = r * (groundtruth[j] - value[j])
        deltaj.append(valuej)
    deltai = []
    for i in range(0,len(hiddennode)):
        f = derivative(hiddennode[i])
        for j in range(0,len(outputnode)):
            valuei = f * outputlist[i][j] * deltaj[j]
        deltai.append(valuei)

    for i in range(0,inputcount):
        for j in range(hiddencount):
            hiddenlist[i][j] = hiddenlist[i][j] + (float(alpha) * float(i) * float(deltai[j]))

    for i in range(best):
        for j in range(4):
            outputlist[i][j] = outputlist[i][j] + (float(alpha) * float(i) * float(deltaj[j]))

def derivative(value):
    if(value <= 0):
        return 0.001
    else:
        return 1

def softmax_function(list):
    list2 = []
    newsum = 0
    maximum_value = max(list)
    for i in range(len(list)):
        list[i] = list[i] - maximum_value
    for i in range(len(list)):
        newsum = newsum + math.e ** list[i]
    for i in list:
        numerator = math.e ** i
        value = float(numerator) / float(newsum)
        list2.append(value)

    return list2


def activation_function(list):
    function_list = []
    for i in range(0,len(list)):
        if (list[i] > 0):
            function_list.append(list[i])
        else:
            value = 0.001 * list[i]
            function_list.append(value)
    return function_list
##################################################Testing#########################################################################################################

def test_neuralnets(file,best):
    input = []
    accuracy = []
    with open(file) as f:
        for line in f:
            countlist = 0
            angle = 0
            input = []
            for word in line.split():
                if (countlist == 1):
                    angle = word
                if (countlist >= 2):
                    input.append(word)
                countlist += 1
            accuracy = hidden_test(input, best, angle, accuracy)

    return accuracy

def hidden_test(input,best,angle,accuracy):
    global hiddenlist
    hiddennode = []
    value = 0
    for i in range(0, best):
        sum = 0
        for j in range(0, len(input)):
            value = float(hiddenlist[j][i]) * float(input[j])
            sum = sum + value
        hiddennode.append(sum)
    count = len(input)
    input_outputlayer = activation_function(hiddennode)
    x = output_test(input_outputlayer, angle, hiddennode, count, best,accuracy)
    return x

def output_test(input,angle,hiddennode,count,best,accuracy):
    global outputlist
    outputnode = []
    for i in range(0,4):
        sum = 0
        for j in range(0,len(input)):
            value = float(outputlist[j][i]) * float(input[j])
            sum = sum + value
        outputnode.append(sum)
    outputs = softmax_function(outputnode)
    valuemax = 0
    for i in range(len(outputs)):
        if(outputs[i] > valuemax):
            valuemax = outputs[i]
            indexmax =  i
    if(indexmax == 0):
        outputval = 0
    elif(indexmax == 1):
        outputval = 90
    elif(indexmax == 2):
        outputval = 180
    else:
        outputval = 270

    print angle
    print outputval
    angle = int(angle)
    if(outputval == angle):
        accuracy.append(1)
    else:
        accuracy.append(0)

    return accuracy

def calculate_accuracy(accuracy):
    true = 0
    suma = 0
    for i in accuracy:
        if(i == 1):
            true += 1
        suma += 1

    tvalue = float(true) / float(suma)
    print ("The accuracy of the neural networks is"), tvalue

######################################################### MAIN FUNCTION ###########################################################################3
#train_file = sys.argv[1]
#test_file = sys.argv[2]
#best = sys.argv[3]
#model_file = sys.argv[4]
best = 192
model_file = 0
method = 'nnet'
list = []
train_file = 'Prerana-test.txt'
test_file = 'test_data.txt'

with open(train_file) as x:
    for line in x:
        count = len(line.split())
        break
if(method == 'nnet'):
    for i in range(count - 2):
        hiddenlist[i] = {}
        for j in range(best):
            weight = numpy.random.normal()
            hiddenlist[i][j] = weight

    for i in range(best):
        outputlist[i] = {}
        for j in range(4):
            weight = numpy.random.normal()
            outputlist[i][j] = weight
    print("Training the Model")
    model = neural_nets(train_file,best)
    print("Testing the Model")
    output = test_neuralnets(test_file,best)
    calculate_accuracy(output)
#with open("model_file.txt", "wb") as fileHandle:
#pickle.dump(model, fileHandle)