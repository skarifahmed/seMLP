import os
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from scipy.io import loadmat
import datetime
import copy
import sys
import statistics as st
from scipy.stats import pearsonr
import json
from core.data_processor import DataLoader
from core.model import Model

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


TESTING = False
# data = "snp500"
data = sys.argv[1]

# TESTING
if TESTING:
    EVOLUTIONS = 3
    NUM_MODELS = 20
    epochs = 100
else:
    EVOLUTIONS = 30
    NUM_MODELS = 100
    epochs = 500

# Parameters
# FITNESS_TYPE = sys.argv[1]
FITNESS_TYPE = "NORMALIZED_CREDITS"
LAYERS_LOWER_LIMIT = int(sys.argv[2])
LAYERS_UPPER_LIMIT = int(sys.argv[3])
NODES_LOWER_LIMIT = 10
NODES_UPPER_LIMIT = 256
# BINARY_DIGITS_LIMIT = '010b' #For 10 digits -> 1024
BINARY_DIGITS_LIMIT = '08b'

NUM_FEATURES = 39
KEEP_PROB = 1
BETA = 10**-4
BATCH_SIZE = 32
learning_rate = 0.001  #Keep learning rate fixed first

# Genetic algo parameters
mut_nodes_rate = 0.001
mut_delta_rate = 0.005
crossover_rate = 0.7




log_name = "log_" + sys.argv[0] + "__" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".txt"
log = open(log_name,'w')


class multi_layer:
    def __init__(self, layers, generation, delta):
        self.generation = generation
        self.layers = layers
        self.delta = delta
        self.fitness = None
        self.chromosome = None
        self.rms = None
        self.ratio = None
        self.pearson = None
        self.prediction = None


def printModel(model):
    # Creates a string for printing to console
    # "Delta1: %g, Delta2: %g layer: %g Model Layer1: %g Layer2: %g" %(model.delta[0], model.delta[1], layer, model.layer1, model.layer2)
    string = "GEN: " + str(model.generation) + " | Num_Layers:" + str(len(model.layers)) + " | DELTA: " + str(model.delta) + " | LAYERS: " + str(model.layers) + " | INITIAL LAYERS: " + str(model.chromosome[1:]) + " | FITNESS: " + str(model.fitness) + " | R2: " + str(model.pearson) + " | RMS: " + str(model.rms) + " | Ratio: " + str(model.ratio)

    return string


def createModel():
    if LAYERS_LOWER_LIMIT != LAYERS_UPPER_LIMIT:
        num_layers = np.random.randint(LAYERS_LOWER_LIMIT, LAYERS_UPPER_LIMIT)
    else:
        num_layers = int(LAYERS_LOWER_LIMIT)
    # print ("Number of layers: ", num_layers)
    layer = np.random.randint(NODES_LOWER_LIMIT, NODES_UPPER_LIMIT)
    delta = []
    for i in range(num_layers):
        gen_delta = np.random.rand()
        while gen_delta == 0:
            gen_delta = np.random.rand()
        delta.append(gen_delta)
    # Calculate nodes in layers
    layers = []
    for k in range(num_layers): 
        single_layer = int(delta[k] * layer)
        layers.append(single_layer)
    model = multi_layer(layers, 1, delta)

    return model, layer 


def createAllModels():
    all_models = []

    for k in range(NUM_MODELS):
        model, layer = createModel()

        # Check if any layers are zero
        i=0
        while i < len(model.layers):
            # print ("Model layer: ", len(model.layers))
            # print ("i: ", i)
            while model.layers[i] == 0:
                model, layer = createModel()
                i=-1
            i+=1

        # Chromosome
        bin_layer = format(layer, BINARY_DIGITS_LIMIT)
        model.chromosome = [model.delta]
        for j in range(len(model.layers)):
            model.chromosome.append(bin_layer)

        print (printModel(model))
        all_models.append(model)

    return all_models


def select(pop):
#    sort based on highest to lowest fitness function
    if len(pop) > 1:
        sorted_models = sorted(pop, key=lambda x:x.fitness, reverse=True)
        index = int(0.1 * len(pop))
        fittest_models = sorted_models[:index]
        print ("In select fittest_models: ", fittest_models, index)

        for model in sorted_models:
            print ("Fitness:", model.fitness, model.layers)
        
        print ("End selection")
        return fittest_models
    else:
        print ("End selection, pop=1")
        return pop



def mutate(pop):
    print ("In mutation", " length of pop:", len(pop))
    for k, model in enumerate(pop):
        if k>(0.1*len(pop)):
            # Save old model
            old_model = copy.copy(model)
            mutated_flag = mutate_single(model)
            if mutated_flag:
                bits = model.chromosome[1:]
                delta = model.delta
                new_layers = calcLayers(bits, delta)
                # For testing
                # new_layers[0] = 0
                print ("Index:", k, "Mutated layers: ", new_layers)
                print ("Prior chromosome: ", model.chromosome)
                i=0
                while i<len(new_layers):
                    while new_layers[i] == 0:
                        print ("inside mutate 0 loop")
                        model = old_model
                        flag = mutate_single(model)
                        if flag:
                            bits = model.chromosome[1:]
                            delta = model.delta
                            new_layers = calcLayers(bits, delta)
                        i=-1
                        print ("after mutate check: ", new_layers)
                    i+=1
                model.layers = new_layers
                print ("after mutate check 2: ", model.layers)
                print ("After chromosome: ", model.chromosome)



    return pop

def mutate_single(model):
    # Mutate delta
    mutated_flag = False
    for i in range(len(model.chromosome[0])):
        prob_delta = np.random.rand()
        # For testing
        # prob_delta = 0.0000001
        # print ("delta before:", model.chromosome[0][i])
        if prob_delta < mut_delta_rate:
            mutated_flag = True
            mutated_delta = np.random.rand()
            model.chromosome[0][i] = mutated_delta
            # print ("delta after:", model.chromosome[0][i])
    # Mutate nodes
    # print ("bit string:", model.chromosome[1:])
    for k, bit_str in enumerate(model.chromosome):
        if k!=0:
            bit_str_copy = ""
            mutated_flag_bit = False
            for j in range(len(bit_str)):
                prob_bit = np.random.rand()
                bit = bit_str[j]
                if prob_bit < mut_nodes_rate:
                    inverted_bit = str(1 - int(bit))
                    bit_str_copy = bit_str_copy + inverted_bit
                    mutated_flag_bit = True
                    mutated_flag = True
                else:
                    bit_str_copy = bit_str_copy + bit
            # print ("bit str copy:", bit_str_copy)
            # print ("model bit str before:", model.chromosome[k])
            if mutated_flag_bit:
                model.chromosome[k] = bit_str_copy
            # print ("model bit str after:", model.chromosome[k])
            # print ("mutated nodes:", model.chromosome[1:])
    return mutated_flag
    


def getProb(pop_len):
# Calculate rank weightings
    ranking = np.arange(1,pop_len + 1)
    denom = np.cumsum(ranking)[-1]
    
    prob = []
    for i in range(pop_len):
        rank = i + 1
        numerator = pop_len - rank + 1
        prob.append(numerator/denom)

    cdf = np.cumsum(prob)

    return cdf

def getParents(pop, odds):
    prob_dad = np.random.rand()
    prob_mum = np.random.rand()

    # print ("Pop: ", pop)
    print ("Odds:", odds)
    print ('dad prob', prob_dad)
    print ('mum prob', prob_mum)
    for j in range(0, len(odds)):
        if j==0:
            lower_bound = 0
        else:
            lower_bound = odds[j-1]
        if prob_dad <= odds[j] and prob_dad > lower_bound:
            print ('dad j:', j)
            dad = pop[j]
            print ('dad', dad.fitness)
        if prob_mum <= odds[j] and prob_mum > lower_bound:
            print ("mum j:", j)
            mum = pop[j]
            print ('mum', mum.fitness)
    return dad, mum

def crossoverDelta(dad, mum, crossed_indexes):
    print ("------------INSIDE CROSSOVER DELTA-----------------")
    # Crossover delta
    x = dad.chromosome[0]
    y = mum.chromosome[0]
    greater = True if len(x)>=len(y) else False
    print ("Greater: ", greater)
    
    
    # delta_prob = np.random.rand()
    

    if greater:
        num = x
        denom = y
        partition = math.floor(len(num) / len(denom))
    else:
        num = y
        denom = x
        partition = math.floor(len(num) / len(denom))

    c1_delta = np.zeros(len(num))
    c2_delta = np.zeros(len(denom))

    print ("c1 delta:", c1_delta)
    print ("c2 delta:", c2_delta)

    w = 0
    for j in range(len(crossed_indexes)):
        beta = np.random.rand()
        print ("BETA: ", beta)
        pos = crossed_indexes[j]
        print ("pos: ", pos)
        # # From start to pos
        # for a in range(j, pos):
        #     c1_delta[a] = num[a]
        # # After pos to end
        # for z in range(pos+1, j+partition):
        #     c1_delta[z] = num[z]
        # For pos
        c1_delta[pos] = (1-beta) * num[pos] + beta * denom[w]
        c2_delta[w] = (1-beta) * denom[w] + beta * num[pos]
        
        # Check for zero delta or delta>1:
        counter = 0
        while c1_delta[pos] == 0 or c1_delta[pos] > 1 or c2_delta[w] == 0 or c2_delta[w] > 1:
            if counter < 10:
                beta = np.random.rand()
                c1_delta[pos] = (1-beta) * num[pos] + beta * denom[w]
                c2_delta[w] = (1-beta) * denom[w] + beta * num[pos]
            else:
                sum = c1_delta[pos] + c2_delta[w]
                c1_delta[pos] = c1_delta[pos] / sum
                c2_delta[w] = c2_delta[w] / sum
        
        # Increment index for smaller child
        w += 1

    # Fill in missing delta:
    for y in range(len(num)):
        if c1_delta[y] == 0:
            c1_delta[y] = num[y]

    print ("c1_delta:", c1_delta)
    print ("c2_delta:", c2_delta)
    return c1_delta, c2_delta

def crossoverNodes(dad, mum):
    bit_prob = np.random.rand()
    print ("-----------IN CROSSOVER NODES:------------")
    d = dad.chromosome[1:]
    m = mum.chromosome[1:]
    greater = True if len(d) >= len(m) else False
    print ("dad chromosome: ", dad.chromosome)
    print ("mum chromosome: ", mum.chromosome)

    if greater:
        num = d
        denom = m
        partition = math.floor(len(num) / len(denom))
    else:
        num = m
        denom = d
        partition = math.floor(len(num) / len(denom))

    c1_bit=["" for i in range(len(num))]
    c2_bit=["" for j in range(len(denom))]
    
    print ("partition: ", partition)
    l = 0
    w = 0
    crossed_indexes = []
    print ("length of num and denom: ", len(num), len(denom) )
    for w in range(len(denom)):
        pos = np.random.randint(l, l+partition)
        print ("pos: ", pos)
        crossed_indexes.append(pos)
        # # From start to pos
        # for j in range(l, pos):
        #     c1_bit[j] = num[j]
        # # After pos to end
        # for z in range(pos+1, l+partition):
        #     print ("z: ", z)
        #     c1_bit[z] = num[z]
        # Crossover at position pos
        k = np.random.randint(1, 9)
        print ("k: ", k)
        print ("num[pos]: ", num[pos])
        print ("denom[w]: ", denom[w])
        c1_bit[pos] = num[pos][:k] + denom[w][k:]
        print ("c1_bit[pos]:", c1_bit[pos])
        c2_bit[w] = denom[w][:k] + num[pos][k:]
        print ("c2_bit[w]:", c2_bit[w])

        l+=partition

    # Fill in missing 
    for y in range(len(num)):
        if c1_bit[y] == "":
            c1_bit[y] = num[y]
    
    for y in range(l, len(num)):
        c1_bit[l] = num[l]

    print ("c1_bit:", c1_bit)
    return c1_bit, c2_bit, crossed_indexes


def crossover(pop, total_pop):
    # pop must be arranged from highest to lowest fitness
    print ("In crossover")
    fittest_length = len(total_pop)
    new_pop = []

    odds = getProb(fittest_length)

    for i in range(int((0.9 * fittest_length)/2)):
        dad, mum = getParents(total_pop, odds)
        while dad == mum:
            dad,mum = getParents(total_pop, odds)

        # Crossover nodes in layer
        c1_bit, c2_bit, crossed_indexes = crossoverNodes(dad, mum)
        
        # Crossover delta
        c1_delta, c2_delta = crossoverDelta(dad, mum, crossed_indexes)


        generation = dad.generation  + 1

        # Child 1
        c1_layers = calcLayers(c1_bit, c1_delta)
        # print ("c1_b0:", c1_b0, "c1_b1:", c1_b1)

        # Child 2
        c2_layers = calcLayers(c2_bit, c2_delta)
        
        # if any of the layers are zero:
        zero_counter = 0

        # For testing
        # c1_layers[1] = 0

        # For child 1
        # z=0
        # while z < len(c1_layers):
        # # for z in range(len(c1_layers)):
        #     while c1_layers[z]==0:
        #         print ("inside zero loop of c1", c1_layers[z])
        #         if zero_counter > 10:
        #             dad, mum = getParents(total_pop, odds)
        #             while dad == mum:
        #                 dad,mum = getParents(total_pop, odds)

        #         c1_bit, c2_bit, crossed_indexes = crossoverNodes(dad, mum)
        #         c1_delta, c2_delta = crossoverDelta(dad, mum, crossed_indexes)
        #         c1_layers = calcLayers(c1_bit, c1_delta)
        #         c2_layers = calcLayers(c2_bit, c2_delta)
        #         zero_counter +=1
        #         z=-1
        #     z+=1

        # Check if layer is zero for both childs
        flag1 = False
        flag2 = False

        while True:
            # Check child 1 zero:
            z=0
            while z < len(c1_layers):
                # print ("z: ", z, len(c1_layers))
                if(c1_layers[z] == 0):
                    print ("inside zero loop of c1")
                    flag1 = True
                    c1_bit, c2_bit, c1_layers, c2_layers = crossoverSingle(dad, mum)
                    z=0
                    break
                if z == len(c1_layers)-1:
                    flag1 = False
                z+=1
            # Check child 2 zero if no zero layers in child 1
            if not flag1:
                y=0
                while y < len(c2_layers):
                    if(c2_layers[y] == 0):
                        flag2 = True
                        c1_bit, c2_bit, c1_layers, c2_layers = crossoverSingle(dad, mum)
                        y=0
                        break
                    if y == len(c2_layers)-1:
                        flag2 = False
                    y+=1
            
            if not flag1 and not flag2:
                break
                
        # Create child models
        child1 = multi_layer(c1_layers, generation, c1_delta)
        child1.chromosome = [c1_delta]
        for c1_layer in c1_bit:
            child1.chromosome.append(c1_layer)

        child2 = multi_layer(c2_layers, generation, c2_delta)
        child2.chromosome = [c2_delta]
        for c2_layer in c2_bit:
            child2.chromosome.append(c2_layer)

        new_pop.append(child1)
        new_pop.append(child2)
    
    # Increase generation for fittest half
    for model in pop:
        model.generation += 1
    joined_pop = pop + new_pop
    print ("joined_pop length:", len(joined_pop))
    print ("End Crossover-----------------")
    return joined_pop


def crossoverSingle(dad, mum):

    print ("Inside crossoverSingle")
    c1_bit, c2_bit, crossed_indexes = crossoverNodes(dad, mum)
    c1_delta, c2_delta = crossoverDelta(dad, mum, crossed_indexes)
    c1_layers = calcLayers(c1_bit, c1_delta)
    c2_layers = calcLayers(c2_bit, c2_delta)

    return c1_bit, c2_bit, c1_layers, c2_layers
  

    
def calcLayers(c_bit, c_delta):
    c_layers = []
    for i in range(len(c_bit)):
        orig_layer = int(c_bit[i], 2)
        c_layers.append(int(orig_layer * c_delta[i]))

    return c_layers

# Build the graph for the deep net
def buildGraph(x, neurons, keep_prob):
    all_weights = []

    h1l_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, neurons[0]], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='h1l_weights')
    h1l_biases  = tf.Variable(tf.zeros([neurons[0]]), name='h1l_biases')
    h1l_perceptron_layer = tf.matmul(x, h1l_weights) + h1l_biases
    h1l_activated = tf.nn.relu(h1l_perceptron_layer)
    h1l_activated = tf.nn.dropout(h1l_activated, keep_prob)
    all_weights.append(h1l_weights)

    activated = h1l_activated
    for i in range(1, len(neurons)):
        weights = tf.Variable(tf.truncated_normal([neurons[i-1], neurons[i]], stddev=1.0/math.sqrt(float(neurons[i-1]))))
        biases  = tf.Variable(tf.zeros([neurons[i]]))
        perceptron_layer = tf.matmul(activated, weights) + biases
        activated = tf.nn.relu(perceptron_layer)
        activated = tf.nn.dropout(activated, keep_prob)
        all_weights.append(weights)

    output_weights = tf.Variable(tf.truncated_normal([neurons[-1], 1], stddev=1.0/math.sqrt(float(neurons[-1]))), name='output_weights')
    output_biases = tf.Variable(tf.zeros(1), name='output_biases')
    output_layer  = tf.matmul(activated, output_weights) + output_biases
    all_weights.append(output_weights)
    # print ("Shape:", h1l_weights.shape)

    print ("Finished building Graph")
    return output_layer, all_weights

def train(model, orig_configs, beta = BETA, batch_size = BATCH_SIZE):
    # parameters--> Train 3 models 
    total_models = []
    for i in range(3):
        model_new = copy.copy(model)
        parameters = {
            "DAYS": i,
            "FROM_END": -(i+1)
        }
        orig_configs['parameters'] = parameters
        configs = orig_configs

        data = DataLoader(
            os.path.join('data', configs['data']['filename']),
            configs['data']['train_test_split'],
            configs['data']['columns'],
            configs['parameters']['DAYS'],
            configs['parameters']['FROM_END']
        )
        

        # Get and reshape data
        trainX, trainY = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
        )

        testX, testY = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )

        testX_un, testY_un = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
        )

        # Get the correct results
        if i == 0:
            correct = testY_un

        print ("trainX shape prior: ", trainX.shape)
        trainX = np.reshape(trainX, [-1, 39])
        print ("trainX shape: ", trainX.shape)
        print ("trainY shape: ", trainY.shape)
        print ("testY shape: ", testY.shape)

        testX = np.reshape(testX, [-1, 39])

        # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, 1])
        keep_prob = tf.placeholder(tf.float32)

    #    Deconstruct the model
        neurons = model.layers

        # Build the hidden layers
        output_layer, all_weights = buildGraph(x, neurons, keep_prob)

        # Add L2 regularizer
        regularizer = 0
        for weight in all_weights:
            regularizer += tf.nn.l2_loss(weight)
        loss = tf.reduce_mean(tf.square(y_ - output_layer))
        loss = tf.reduce_mean(loss + beta * regularizer)
        # loss = tf.sqrt(tf.reduce_mean(loss + beta * regularizer))

        # Create the Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
        train_acc = []
        test_err = []

        # Mean Square Error
        error = tf.sqrt(tf.reduce_mean(tf.square(y_ - output_layer)))
        # error = tf.reduce_mean(tf.square(y_ - output_layer))
        prediction = output_layer
        
        trainX_local=trainX
        # trainY_local=trainY
        trainY_local = trainY


        total_test_errs = []

        with tf.Session() as sess:   #Maybe change this to outside the folds loop
            sess.run(tf.global_variables_initializer()) 
            
            for i in range(epochs):
                start = 0
                end = batch_size
                for k in range(int(trainX.shape[0]/batch_size)):
                    if k!=0:
                        start += batch_size
                        end += batch_size
                        if end  >= trainX.shape[0]:
                            end = trainX.shape[0]
                    trainX_ = trainX_local[start:end]
                    trainY_ = trainY_local[start:end]
                    train_op.run(feed_dict={x: trainX_, y_: trainY_, keep_prob: KEEP_PROB})
                if i % 100 == 0 or i==(epochs-1):
                    predicted_output = prediction.eval(feed_dict={x: testX, y_:testY, keep_prob: 1})
                    print('Num_layers: %s, Layers : %s  epoch %d: training error: %g, test error: %g'%(str(len(neurons)), str(neurons), i, loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: KEEP_PROB}), error.eval(feed_dict={x: testX, y_: testY, keep_prob: 1})))
                test_err.append(error.eval(feed_dict={x: testX, y_: testY, keep_prob: 1}))
                        
                        

            train_acc.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: KEEP_PROB}))        
            testing_acc = error.eval(feed_dict={x: testX, y_: testY, keep_prob: 1})
            # model.rms = testing_acc
            print ("Final model fitness:", model.fitness, "Delta:", model.delta, "RMS: ", testing_acc)
            predicted_output = prediction.eval(feed_dict={x: testX, y_: testY, keep_prob: 1})

            denorm_y = data.de_normalise_windows(predicted_output)
            model_new.predictions = denorm_y

            # Get Pearson's R2
            corr_relative, p_value = pearsonr(testY.flatten(), predicted_output.flatten())
            corr_real, p_value = pearsonr(testY_un.flatten(), denorm_y.flatten())
            model_new.pearson = corr_real**2
            print ("Model R2: ", model_new.pearson)

            # Save model
            total_models.append(model_new)

        # Reset Graphs
        sess.close()
        tf.reset_default_graph()

    
    print ("Total models : ", len(total_models))
    model_predictions = [model.predictions for model in total_models]
    print ("Model_predictions: ", len(model_predictions))

    print ("Correct: ", len(correct))
    print ("t1: ", len(model_predictions[0]), model_predictions[0][-2], model_predictions[0][-1])
    print ("t2: ", len(model_predictions[1]), model_predictions[1][-2], model_predictions[1][-1])
    print ("t3: ", len(model_predictions[2]), model_predictions[2][-2], model_predictions[2][-1])

    predicted = []
    k=0
    while k<len(correct)-1:
        # print ("i: ", i)
        predicted.append(model_predictions[0][k])
        predicted.append(model_predictions[1][k])
        predicted.append(model_predictions[2][k])
        k+=3
    correct = correct.flatten()
    # Remove extra value in correct 
    correct = correct[:-1]
    predicted = np.array(predicted).flatten()
    print ("Predicted: ", len(predicted), predicted[-1])

    corr, p_value = pearsonr(correct, predicted)
    print ("R2: ", corr**2)

    i=0
    total_moving_c = []
    total_moving_p = []
    while i<len(correct)-3:
        sum_c = correct[i] + correct[i+1] + correct[i+2]
        sum_p = predicted[i] + predicted[i+1] + predicted[i+2]
        moving_c = sum_c/3
        moving_p = sum_p/3
        total_moving_c.append(moving_c)
        total_moving_p.append(moving_p)
        i+=3
    print ("Moving C: ", len(total_moving_c), total_moving_c[:5])
    print ("Moving P: ", len(total_moving_p), total_moving_p[:5])
    print ("Correct: ", correct[:15])

    diff = np.array(total_moving_c)-np.array(total_moving_p)
    print ("Diff: ", diff[:20])
    diff = abs(diff)
    avg = np.cumsum(diff)[-1] / len(diff)
    print ("Average Diff: ", avg)

    # R2 for MA
    corr_ma, p_value = pearsonr(total_moving_c, total_moving_p)
    print ("R2 for MA: ", corr_ma**2)
    
    # Assign R2 to model
    model.pearson = corr_ma**2
    model.rms = avg

    


def calcFitness(model, testing_acc):
    if FITNESS_TYPE == "PURE_ACC_VARY_LAYERS":
        fitness = 1/testing_acc
    elif FITNESS_TYPE == "LOG10_RATIO_VARY_LAYERS":
        sum_ratio = 0
        for i in range(1, len(model.layers)):
            sum_ratio += model.layers[i-1] / model.layers[i]
        layers_diff = len(model.layers) - 1 if len(model.layers) > 1 else len(model.layers)
        mean_ratio = sum_ratio/layers_diff
        mean_ratio = math.log10(mean_ratio)
        fitness = mean_ratio / testing_acc

    return fitness


def calcPearsonFitness(pop):

    if FITNESS_TYPE == "RMS_MEAN":
        rms_sum = 0
        # Get total sum of RMS
        for model in pop:
            rms_sum += model.rms
        rms_mean = rms_sum / len(pop)
        for model in pop:
            ratio = model.layer1 / model.layer2
            fitness = ratio / (model.rms / rms_mean)
            model.fitness = fitness

    elif FITNESS_TYPE == "LOG_RMS_MEAN":
        rms_sum = 0
        # Get total sum of RMS
        for model in pop:
            rms_sum += model.rms
        rms_mean = rms_sum / len(pop)
        for model in pop:
            ratio = model.layer1 / model.layer2
            fitness = math.log(ratio) / (model.rms / rms_mean)
            model.fitness = fitness

    elif FITNESS_TYPE == "NORMALIZED_CREDITS":
        # Calculate offset for numerator 
        if LAYERS_LOWER_LIMIT != LAYERS_UPPER_LIMIT:
            denom_offset = (1 - ((LAYERS_UPPER_LIMIT-1) * 0.1)) / (LAYERS_UPPER_LIMIT - 2)
            print ("Denom Offset: ", denom_offset)
            num_offset = [0.225]
            for i in range(LAYERS_LOWER_LIMIT, LAYERS_UPPER_LIMIT):
                if i>1: 
                    offset = (denom_offset * (i-1)) + (i * 0.1)
                    num_offset.append(offset)
            print ("Offset array: ", num_offset)
        else:
            denom_offset = (1 - (LAYERS_LOWER_LIMIT * 0.1)) / (LAYERS_LOWER_LIMIT - 1)
            print ("Denom Offset: ", denom_offset)
            num_offset = [1]

        # Calculate denom=normalized rms
        total_pearson = [pop[i].pearson for i in range(len(pop))]
        print ("Total pearson array: ", total_pearson)
        min_pearson = np.min(total_pearson, axis=0)
        max_pearson = np.max(total_pearson, axis=0)
        print ("minimum: ", min_pearson)
        print ("max: ", max_pearson)
        print ("Median: ", st.median(total_pearson))
        # print ("Mean: ", st.mean(total_rms))

        # Calculate IQR
        total_pearson, max_pearson = calcIQR(total_pearson, max_pearson)

        # Calculate numerator=normalized credits
        min_fitness = denom_offset / (denom_offset + 1)
        print ("min fitness: ", min_fitness)
        pearson_penalty = ((denom_offset +1) / min_fitness) - (denom_offset + 1)
        print ("pearson_penalty: ", pearson_penalty)
        for model in pop:
            if model.pearson <= max_pearson:
                denom = scale(model.pearson, min_pearson, max_pearson)
            else:
                denom = 1 + pearson_penalty
            intervals = len(model.layers) - 1 if (len(model.layers) > 1) else 1
            print ("Intervals: ", intervals, "Layers: ", model.layers)
            credits = 0
            layer_penalty = 0
            if (len(model.layers) > 1):
                for i in range(1, len(model.layers)):
                    interval_ratio = model.layers[i-1] / model.layers[i]
                    if interval_ratio > 1:
                        credits += 1
            layer_penalty = len(model.layers) * 0.1
            # numerator = (credits - layer_penalty + 1) / intervals
            num_final_offset = num_offset[len(model.layers) - 1] if LAYERS_LOWER_LIMIT!=LAYERS_UPPER_LIMIT else 1
            print ("num_final_offset: ", num_final_offset)
            numerator = (credits - layer_penalty + num_final_offset) / intervals
            print ("Numerator: ", numerator)
            denom += denom_offset
            print ("Offset: ", denom_offset)
            print ("Denom: ", denom)
            fitness = numerator * denom
            print ("Indiv fitness: ", fitness)
            print ("----------------------------------")
            model.fitness = fitness
            model.ratio = numerator

def calcIQR(total_rms, max_rms):
    q75 = np.percentile(total_rms, 75)
    q25 = np.percentile(total_rms, 25)
    print ("q75, q25: ", q75, q25)
    iqr = q75 - q25
    cut_off = 6 * iqr
    upper = cut_off + q75
    print ("Upper: ", upper)
    outliers = []
    outliers_index = []
    for index, rms in enumerate(total_rms):
        if rms > upper:
            outliers.append(rms)
            outliers_index.append(index)
    print ("Outliers: ", outliers)
    print ("Outliers index: ", outliers_index)
    new_total_rms = []
    if len(outliers_index)>0:
        for k, rms in enumerate(total_rms):
            if k not in outliers_index:
                new_total_rms.append(rms)
        total_rms = new_total_rms
        print ("New total_rms: ", total_rms)
        print ("New total_rms length: ", len(total_rms))
        max_rms = np.max(total_rms, axis=0)
        print ("New max: ", max_rms)

    return total_rms, max_rms



def saveModels(pop, evolution):
    current_gen = []
    index = 0
    for model in pop:
        index += 1
        # current_model = [model.layer1, model.layer2, model.fitness, model.chromosome, model.generation]
        # current_model = str(index) + ". " + "Generation " + str(model.generation) + ", " + "Layer 1: " + str(model.layer1) + ", " + "Layer 2: " + str(model.layer2) + ", " + "Fitness: " + str(model.fitness) + ", " + "RMS: " + str(model.rms) + ", " + "Chromosome: " + str(model.chromosome)
        # current_gen.append(current_model)
        current_model =  str(index) + ". " + printModel(model)
        current_gen.append(current_model)
    log.write('\n' + '-------------------------------' + 'EVOLUTION: ' + str(evolution) + '------------------------------' + '\n')
    log.write('\n'.join(current_gen))

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    all_models = createAllModels()
    print ("Length all_models:", len(all_models))
    fittest = 0
    counter = 0
    best_fitness = []
    best_pearson = []
    best_rms = []
    mean_fitness = []
    mean_rms = []
    mean_ratio = []
    mean_pearson = []

    for k in range(len(all_models)):
        print ("---------------------------------------")
        print ("Generation:", all_models[k].generation, "Number:", k+1, "/", len(all_models))
        train(all_models[k], configs)
    saveModels(all_models, 0)
    calcPearsonFitness(all_models)


#  GA Evolution
    for k in range(EVOLUTIONS):
        if k==0:
            pop = all_models
        print ("Evolution Stage:", k+1)
       
        total_pop = pop
        fittest_pop = select(pop)

        print ("Fittest_pop: ", fittest_pop)
  
        
        pop = crossover(fittest_pop, total_pop)
        pop = mutate(pop)
        start = int(0.1 * len(pop))
        for j in range(start, len(pop)):
            print ("---------------------------------------")
            print ("Generation:", pop[j].generation, "Number:", j+1, "/", len(pop))
            train(pop[j], configs)
        # Calculate Fitness
        calcPearsonFitness(pop)

        # Sort in Descending Order
        pop.sort(key=lambda x: x.fitness, reverse=True)

        # Logging
        saveModels(pop, k+1)

        # Best
        best_fitness.append(pop[0].fitness)  
        best_rms.append(pop[0].rms)
        best_pearson.append(pop[0].pearson)

        # Avg fitness & rms
        sum_fitness = 0
        sum_rms = 0
        sum_ratio = 0
        sum_pearson = 0
        total_rms = []
        total_pearson = []
        for model in pop:
            # print ("model.fitness: ", model.fitness)
            sum_fitness += model.fitness
            sum_ratio += model.ratio
            sum_pearson += model.pearson
            total_rms.append(model.rms)
            total_pearson.append(model.pearson)
        mean_fitness.append(sum_fitness / len(pop))
        mean_ratio.append(sum_ratio/len(pop))
        mean_pearson.append(sum_pearson/len(pop))
        # mean_rms.append(sum_rms / len(pop))
        total_rms, new_max = calcIQR(total_rms, 0)
        for rms in total_rms:
            sum_rms += rms
        mean_rms.append(sum_rms / len(total_rms))
        print ("New rms length in Main: ", len(total_rms))

        # Printing
        index = 0
        print ("--------------------------------------------Models in Evolution: ", k+1 , "--------------------------")
        for model in pop:
            index += 1
            # current_model = str(index) + ". " + "Generation " + str(model.generation) + ", " + "Layer 1: " + str(model.layer1) + ", " + "Layer 2: " + str(model.layer2) + ", " + "Fitness: " + str(model.fitness) + ", " + "RMS: " + str(model.rms) + ", " + "Chromosome: " + str(model.chromosome)
            print (str(index) + ". " + printModel(model))
        print ("--------------------------------------------------------------------------------------------------------")

    # model = pop[0]
    # fittest = "Generation " + str(model.generation) + ", " + "Layer 1: " + str(model.layer1) + ", " + "Layer 2: " + str(model.layer2) + ", " + "Chromosome: " + str(model.chromosome)
    # pop.sort(key=lambda x: x.fitness, reverse=True)
    print ("-----------------------------------------------------FINAL LIST--------------------------------------------------")
    for model in pop:
        print(printModel(model))
    fittest = printModel(pop[0])
    log.write('\n' + '\n' + "Fittest: " + str(fittest))
    log.write('\n' + '\n' + "Best Fitness: " + str(best_fitness))
    log.write('\n' + '\n' + "Best Pearson: " + str(best_pearson))
    log.write('\n' + '\n' + "Best RMS: " + str(best_rms))
    log.write('\n' + '\n' + "Average Fitness: " + str(mean_fitness))
    log.write('\n' + '\n' + "Average Pearson: " + str(mean_pearson))
    log.write('\n' + '\n' + "Average RMS: " + str(mean_rms))
    log.write('\n' + '\n' + "Average Ratio: " + str(mean_ratio))
    log.write('\n' + '\n' + "Fitness Type: " + FITNESS_TYPE)
    log.write('\n' + '\n' + "Data: " + data)
    print("Best:", printModel(pop[0]))
    print ("Best Fitness:", best_fitness)
    print ("Best Pearson:", best_pearson)
    print ("Best RMS:", best_rms)
    print ("Avg Fitness:", mean_fitness)
    print ("Avg Pearson:", mean_pearson)
    print ("Avg RMS: ", mean_rms)
    print ("Avg Ratio: ", mean_ratio)
    print ("Fitness Type: ", FITNESS_TYPE)
    print ("Log file name: ", log_name)
    print ("Data: ", data)
    

if __name__ == '__main__':
  main()

