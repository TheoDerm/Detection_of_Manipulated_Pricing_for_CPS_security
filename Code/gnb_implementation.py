import math
import collections
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# Function to return the prior probabitlity of each classes of labels(1 or 0)
# Formula: prior probability = num. of instances of class c / total num. of instances in the dataset
# Returns  a list
def find_prior_probability(label_instances):
    c_counter = collections.Counter(label_instances)
    p_prob = np.ones(2)

    for x in range(0,2):
        p_prob[x] = c_counter[x]/label_instances.shape[0]
    
    return p_prob

# Function to return the variance and the mean of the guideline features in the train data set
# Returns two lists
def find_variance_and_mean(label_set, train_set):
    mean = np.ones((2, train_set.shape[1]))
    variance = np.ones( (2, train_set.shape[1]) )

    c_0 = collections.Counter(label_set)[0]
    num_of_features = train_set.shape[0]

    l_1 = np.ones((num_of_features - c_0, train_set.shape[1]))
    l_0 = np.ones(( c_0, train_set.shape[1]))
    
    # The guidelines labeled with 1 are added to the l_1 array
    # The guidelines labeled with 0 are added to the l_0 array
    
    # Start for l_1
    index = 0
    for x in range(0, num_of_features):
        if (label_set[x] == 1):
            l_1[index] = train_set[x]
            index += 1

    # Start for l_0
    index = 0
    for x in range(0, num_of_features):
        if (label_set[x] == 0):
            l_0[index] = train_set[x]
            index += 1

    
    # Compute the mean and the variance of every feature in the guidelines
    for i in range (0, train_set.shape[1]):
        mean[1][i] = np.mean(l_1.T[i])
        variance[1][i] = np.var(l_1.T[i]) * ((num_of_features-c_0)/((num_of_features-c_0) -1))
       
        mean[0][i] = np.mean(l_0.T[i])
        variance[0][i] = np.var(l_0.T[i]) * (c_0/(c_0 - 1))

    return variance, mean

# Function to return the posterior probabilities of the test data for every class
# Formula: posterior probability(x|c) = P(x1|c) * P(x2|c) * ... * P(xn|c)
# Returns a list
def find_posterior_prob(variance, mean, test_data):
    post_prob = np.ones(2)
    num_of_features = len(mean[1])

    for x in range (0,2):
        prod = 1

        for y in range (0, num_of_features):
            prod = prod * (1/math.sqrt(2 * 3.14 * variance[x][y])) * math.exp(-0.5 * pow((test_data[y] - mean[x][y]),2) / variance[x][y])
        post_prob[x] = prod
    
    return post_prob
    

# Function to to combine the previous 3 functions and implement the Gaussian Naive Bayes algorithm. It computes the conditional
# probability of each class given the test data.
# Returns a list
def gnb_classifier(test_data, train_set, label_set):
    variance , mean = find_variance_and_mean(label_set, train_set)
    prior_probability = find_prior_probability(label_set)
    results = []
    for row in range (len(test_data[:, 0])):
        temp = test_data[row]
        posterior_prob = find_posterior_prob(variance, mean, temp)
        conditional_prob = np.ones(2)
        total_prob = 0

        for x in range (0,2):
            total_prob = total_prob + (posterior_prob[x] * prior_probability[x])
        
        for x in range (0,2):
            conditional_prob[x] = (posterior_prob[x] * prior_probability[x]) / total_prob
        
        prediction = int(conditional_prob.argmax())
        results.append(prediction)
    return results


# Start the actual execution

# Model training:

# Load the data from 'TrainingData.txt' into a dataframe
df = pd.read_csv('TrainingData.txt',sep=',', header=None)

# Create two data sets
train_set = df.iloc[:,:24]              #data set for the guideline prices
labels_set = df.iloc[:,-1]              #data set for the labels of the guideline prices

# Creating 4 new data sets for training and validation.  
# guideline_train = 80% of guideline_price_set,   labels_train = 80% of labels_set
# guideline_test = 20& of guideline_price_set,   labels_test = 20% of labels_set

guideline_train = train_set.head(8000)
labels_train = labels_set.head(8000)

guideline_test = train_set.tail(2000)
labels_test = labels_set.tail(2000)

# Convert the dataframes into correct format
guideline_test = np.asarray(guideline_test)
guideline_train = np.asarray(guideline_train)

labels_test = np.asarray(labels_test)
labels_train = np.asarray(labels_train)

train_set = np.asarray(train_set)
labels_set = np.asarray(labels_set)

# Train the model
train_outcome = gnb_classifier(train_set ,train_set, labels_set)
accuracy = accuracy_score(labels_set, train_outcome)
print ('The accuracy of the model on training is: ', accuracy )

# Validate the model
validation_outcome = gnb_classifier(guideline_test, guideline_train, labels_train)
validation_accuracy = accuracy_score(labels_test, validation_outcome)
print ('The accuracy of the model on validation is: ', validation_accuracy )


# Model prediction:

# Load the data from 'TestingData.txt' into a dataframe
final_data = pd.read_csv('TestingData.txt', sep = ',', header = None)
final_data = np.asarray(final_data)

output_data = np.ones((100,25))
final_outcome = gnb_classifier(final_data,train_set,labels_set)

sum = 0 #keep count of how many instances of tampered guideline pricing curves we have

for i in range(len(final_data)):
    output_data[i] = np.append(final_data[i], final_outcome[i])
    if(final_outcome[i] == 1):
        sum += 1

style = (['%.15g'] * 24) + ['%i']
np.savetxt('TestingResults.txt', output_data, delimiter=',', fmt=style)
print("Predictions saved in the 'TestingResults.txt' file.\nNumber of guidelines pricing curves predcited as abnormal is:", sum)