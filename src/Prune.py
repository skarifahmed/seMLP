#Vary all nodes and layers
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp
from scipy.io import loadmat
import datetime
import copy
from scipy.stats import pearsonr

tf.reset_default_graph()


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# TESTING
epochs = 100


info = "snp500"


BETA = 10**-9
BATCH_SIZE = 128



#Import Dataset
M = loadmat('MackeyGlass.mat')
inputs = M['mackeyglass_input']
targets = M['mackeyglass_target']
trainX = inputs[:3000]
trainY = targets[:3000]
testX = inputs[3000:]
testY = targets[3000:]
NUM_FEATURES = 4


learning_rate = 0.001

class multi_layer:
    def __init__(self, layers, generation, delta):
        self.generation = generation
        self.layers = layers
        self.delta = delta
        self.fitness = None
        self.chromosome = None
        self.rms = None
        self.r2 = None
        self.counter = None


# Build the graph for the deep net
def buildGraph(x, neurons):
    all_weights = []

    h1l_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, neurons[0]], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='h1l_weights')
    h1l_biases  = tf.Variable(tf.zeros([neurons[0]]), name='h1l_biases')
    h1l_perceptron_layer = tf.matmul(x, h1l_weights) + h1l_biases
    h1l_activated = tf.nn.relu(h1l_perceptron_layer)
    all_weights.append(h1l_weights)

    activated = h1l_activated
    for i in range(1, len(neurons)):
        weights = tf.Variable(tf.truncated_normal([neurons[i-1], neurons[i]], stddev=1.0/math.sqrt(float(neurons[i-1]))))
        biases  = tf.Variable(tf.zeros([neurons[i]]))
        perceptron_layer = tf.matmul(activated, weights) + biases
        activated = tf.nn.relu(perceptron_layer)
        all_weights.append(weights)

    output_weights = tf.Variable(tf.truncated_normal([neurons[-1], 1], stddev=1.0/math.sqrt(float(neurons[-1]))), name='output_weights')
    output_biases = tf.Variable(tf.zeros(1), name='output_biases')
    output_layer  = tf.matmul(activated, output_weights) + output_biases
    all_weights.append(output_weights)
    # print ("Shape:", h1l_weights.shape)
    return output_layer, all_weights

def train(model, threshold, beta = BETA, batch_size = BATCH_SIZE):
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, 1])

#    Deconstruct the model
    neurons = model.layers

    # Build the hidden layers
    output_layer, all_weights = buildGraph(x, neurons)

    # Add L2 regularizer
    regularizer = 0
    for weight in all_weights:
        regularizer += tf.nn.l2_loss(weight)
    loss = tf.reduce_mean(tf.square(y_ - output_layer))
    loss = tf.sqrt(tf.reduce_mean(loss + beta * regularizer))

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
    trainY_local=trainY

    # Op to save the variables
    saver = tf.train.Saver()



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        
        # for i in range(epochs):
        #     start = 0
        #     end = batch_size
        #     for k in range(int(trainX.shape[0]/batch_size)):
        #         if k!=0:
        #             start += batch_size
        #             end += batch_size
        #             if end  >= trainX.shape[0]:
        #                 end = trainX.shape[0]
        #         trainX_ = trainX_local[start:end]
        #         trainY_ = trainY_local[start:end]
        #         train_op.run(feed_dict={x: trainX_, y_: trainY_})
        #     if i % 100 == 0 or i==(epochs-1):
        #         print('Num_layers: %s, Layers : %s  epoch %d: training error: %g, test error: %g'%(str(len(neurons)), str(neurons), i, loss.eval(feed_dict={x: trainX, y_: trainY}), error.eval(feed_dict={x: testX, y_: testY})))
        #     test_err.append(error.eval(feed_dict={x: testX, y_: testY}))
                
                

        # train_acc.append(loss.eval(feed_dict={x: trainX, y_: trainY}))        
        # testing_acc = error.eval(feed_dict={x: testX, y_: testY})
        # model.rms = testing_acc
        # print ("Final model fitness:", model.fitness, "Delta:", model.delta, "RMS: ", testing_acc)
        # predicted_output = prediction.eval(feed_dict={x: testX, y_:testY})
        # corr, p_value = pearsonr(testY, predicted_output)

        # Restore variables from disk
        # saver.restore(sess, "saved/model_4.ckpt")
        print ("Restoring")
        saver.restore(sess, "GA_MLP_nasdaq/saved_models/0/0_1__90_51_183_/1.ckpt")
        print ("Model restored")
        rmse_prior = error.eval(feed_dict={x: testX, y_: testY})
        predicted_output_prior = prediction.eval(feed_dict={x: testX, y_:testY})
        corr_prior, p_value = pearsonr(testY, predicted_output_prior)
        r2_prior = corr_prior**2

        # Get desired layers for culling
        names = []
        for counter, v in enumerate(tf.trainable_variables()):
            if counter%2 == 0 and "output" not in v.name:
                names.append(v.name)

        weights = []
        counter = 0
        total_links = 0
        # # #### Changing the weights all layers:
        for index, name in enumerate(names):
            weight_tensor = sess.graph.get_tensor_by_name(name)
            weight_array = sess.run(weight_tensor)
            abs_weights = np.abs(weight_array)
            total_links += weight_array.size
            # Cull the links
            # cull_value = threshold[index]
            if threshold[index] != 0:
                print ("Threshold in train: ", name, threshold[index])
                cull_value = np.percentile(abs_weights, threshold[index])
                for i in range(len(weight_array)):
                    for j in range(len(weight_array[0])):
                        if abs_weights[i][j] <= cull_value:
                            weight_array[i][j] = 0
                assign_op = tf.assign(weight_tensor, weight_array)
                weight_after = sess.run(assign_op)
                # print ("After: ", weight_after)
                # print ("Size of weights: ", weight_after.size)                
                for i in weight_after:
                    for j in i:
                        if j == 0:
                            counter +=1
                weights.append(weight_after)
        # print ("Counter: ", counter)
        model.chromosome = weights
        model.counter = counter
        

        rmse_after = error.eval(feed_dict={x: testX, y_: testY})
        model.rms = rmse_after
        print ("RMSE prior: ", rmse_prior)
        print ("New RMSE ", rmse_after)
        print ("Diff in RMSE: ", rmse_after - rmse_prior)

        predicted_output_after = prediction.eval(feed_dict={x: testX, y_:testY})
        corr_after, p_value = pearsonr(testY, predicted_output_after)
        r2_after = corr_after**2
        model.r2 = r2_after
        print ("Corr prior: ", r2_prior)
        print ("Corr after: ", r2_after)
        print ("Diff in Corr: ", r2_prior - r2_after)

        
        
    sess.close()
    tf.reset_default_graph()

    return model

def directedSearch(model, layer_no, value, low, high):
    # Value is the rms error 
    num_layers = len(model.layers)
    threshold = [0] * num_layers

    mid = (high+low) // 2

    # Error
    if high < low:
        print ("High < Low ERROR", low, high)
        return None

    # Stopping Condition
    if low == high:
        return model, mid-1

    print ("Mid: ", mid)
    print ("Low, High: ", low, high)
    print ("Value: ", value)
    threshold[layer_no] = mid
    model = train(model, threshold)
    print ("---------------------")
    # Record best rmse 
    if model.rms < value:
        value = model.rms

    # Modified Binary Search
    if (model.rms > value):
        print (">")
        return directedSearch(model, layer_no, value, low, mid)

    elif (model.rms <= value):
        print ("<")
        return directedSearch(model, layer_no, value, mid+1, high)



def main():
    # layers = [20, 20]
    # layers = [50,50,50]
    # layers = [184, 124, 106, 85]
    layers = [90,51,183]
    # layers = [28, 45]
    model = multi_layer(layers, 1, None)

    threshold = [0] * len(layers)
    model = train(model, threshold)
    initial_rmse = model.rms

    print("Initial model rms: ", initial_rmse)

    threshold = [0] * len(layers)

    for i in range(len(layers)):
        best_model, threshold[i] = directedSearch(model, i, initial_rmse, 1, 100)

    print ("Final model: -------------------")
    best_model = train(model, threshold)
    print ("Num links culled: ", best_model.counter)
    
    print ("Threshold: ", threshold)

  




    
    # plt.figure(1)
    # plt.plot(range(len(testY)), testY)
    # plt.plot(range(len(predicted_output)), predicted_output)
    # legends=["Actual", "Predicted"]
    # plt.legend(legends, loc="lower right")
    # plt.xlabel("Duration")
    # plt.ylabel('Mackey')
    # plt.title('Dataset')
    

    # plt.figure(2)
    # plt.plot(range(epochs), test_err)
    # plt.xlabel(str(epochs) + ' epochs')
    # plt.ylabel('Test error')
    # plt.title('Epochs vs Test error')

    # plt.figure(3)
    # plt.plot(range(len(targets)), targets)
    # plt.xlabel('Years')
    # plt.ylabel('Price')
    # plt.title('Entire Range')

    

    # plt.show()
  



if __name__ == '__main__':
  main()

