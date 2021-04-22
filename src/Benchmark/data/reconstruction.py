import pickle
import numpy as np 
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

filename = 't1'
with open(filename, 'rb') as f:
    t1 = pickle.load(f)

with open('t2', 'rb') as f:
    t2 = pickle.load(f)

with open('t3', 'rb') as f:
    t3 = pickle.load(f)

with open('correct', 'rb') as f:
    correct = pickle.load(f)

print ("Correct: ", len(correct))
print ("t1: ", len(t1), t1[-2])
print ("t2: ", len(t2), t2[-2])
print ("t3: ", len(t3), t3[-2])

predicted = []

i=0
while i<len(correct)-1:
    # print ("i: ", i)
    predicted.append(t1[i])
    predicted.append(t2[i])
    predicted.append(t3[i])
    i+=3

correct = correct.flatten()
# Remove extra value in correct 
correct = correct[:-1]
predicted = np.array(predicted).flatten()

print ("Predicted: ", len(predicted), predicted[-1])




corr, p_value = pearsonr(correct, predicted)
print ("R2: ", corr**2)

# plt.plot(range(len(correct)), correct, label="Correct")
# plt.plot(range(len(t1.flatten())), t1.flatten(), label="Predicted")
# plt.legend(loc="best")

# plt.show()

# Calculate moving average
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

# Plot moving averages
plt.plot(range(len(total_moving_c)), total_moving_c, label="Correct MA")
plt.plot(range(len(total_moving_p)), total_moving_p, label="Predicted MA")
plt.legend(loc="best")
plt.show()