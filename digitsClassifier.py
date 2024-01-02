# Imports
# mnist digit dataset
from sklearn.datasets import load_digits
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import random

# Function I had from MATLAB
def computeAspectRatio(image, printFlag=False):
    # Computes the aspect ratio of the digit in an image
    # It works because the pixels are either black or white, so the boundaries are clean
    num_rows, num_cols = image.shape
    
    # Initialize the minimum and maximum dimensions
    min_width = num_cols // 2
    max_width = num_cols // 2
    min_height = num_rows // 2
    max_height = num_rows // 2
    
    # For each pixel that is not black, update the min and max dimensions
    for imrow in range(num_rows):
        for imcol in range(num_cols):
            if image[imrow, imcol] != 0:
                min_height = min(min_height, imrow)
                max_height = max(max_height, imrow)
                
                min_width = min(min_width, imcol)
                max_width = max(max_width, imcol)

    width = max_width - min_width
    height = max_height - min_height
    a_ratio = width / height

    # Does not show image by default
    if printFlag:
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Sample with aspect ratio rectangle')
        plt.imshow(image, cmap='gray')
        plt.title('Sample with aspect ratio rectangle')
        plt.gca().add_patch(plt.Rectangle((min_width, min_height), width, height, edgecolor='r', linewidth=2, fill=False))
        plt.show()

    return a_ratio

############################################## 1 - Dataset manipulation: ###############################################

# Load the digits dataset
digits = load_digits()

# Extract the data and labels
# Bunch objects: dictionary-like containers
data = digits['data']
labels = digits['target']

# Filter the data to only include digits with labels 0 and 1, this is the classification problem
filtered_data = data[(labels == 0) | (labels == 1)]
filtered_labels = labels[(labels == 0) | (labels == 1)]

# print(len(filtered_data))

# Randomly selecting five indices from the filtered dataset
random_indices = random.sample(range(len(filtered_data)), 5)

# Plotting the five randomly selected images with their labels
fig = plt.figure(figsize=(10, 2))
fig.canvas.manager.set_window_title('Random samples of 0 and 1')
for i, idx in enumerate(random_indices, 1):
    plt.subplot(1, 5, i)
    plt.imshow(filtered_data[idx].reshape(8, 8), cmap='gray')
    plt.title(f'Label: {filtered_labels[idx]}')
    plt.axis('off')

plt.show()

############################################# 2 - Aspect ratio calculations: ############################################

# Keep arrays for train/test images of datasets D1 and D0
# Initialize lists to store aspect ratios for each digit
aspect_ratios_0 = []
aspect_ratios_1 = []

# Test computation of aspect ratio with output
print("\nAspect ratio of a digit 0: " + str(computeAspectRatio(filtered_data[0].reshape(8, 8),True)))
print("Aspect ratio of a digit 1: " + str(computeAspectRatio(filtered_data[9].reshape(8, 8),True)))

# Iterate through the filtered dataset computing the aspect ratios in the proccess
for img, label in zip(filtered_data, filtered_labels):
    # Compute aspect ratio
    aspect_ratio = computeAspectRatio(img.reshape(8, 8))

    # Store aspect ratio in corresponding list
    if label == 0:
        aspect_ratios_0.append(aspect_ratio)
    elif label == 1:
        aspect_ratios_1.append(aspect_ratio)

# Split the aspect ratio lists into train and test sets
# Update: Changed to have the complete set as training set and the second half as test set
# (because with first half, std of D0 class was 0 which caused problems)

# For digit 0
half_index_0 = len(aspect_ratios_0) // 2
#aspect_ratios_0_train = aspect_ratios_0[:half_index_0]
aspect_ratios_0_test = aspect_ratios_0[half_index_0:]

# For digit 1
half_index_1 = len(aspect_ratios_1) // 2
#aspect_ratios_1_train = aspect_ratios_1[:half_index_1]
aspect_ratios_1_test = aspect_ratios_1[half_index_1:]

# Change explained in the above comments
aspect_ratios_0_train = aspect_ratios_0
aspect_ratios_1_train = aspect_ratios_1

# Compute average aspect ratio for each digit
avg_aspect_ratio_0_train = np.mean(aspect_ratios_0_train)
avg_aspect_ratio_1_train = np.mean(aspect_ratios_1_train)

print("\nAverage Aspect Ratio for Digit 0:", avg_aspect_ratio_0_train)
print("Average Aspect Ratio for Digit 1:", avg_aspect_ratio_1_train)

# distribution of aspect ratios for show
fig = plt.figure(figsize=(10, 5))
fig.canvas.manager.set_window_title('Distribution of Aspect Ratios for Digits 0 and 1')
plt.hist(aspect_ratios_0_train, bins=10, alpha=0.5, label='Digit 0')
plt.hist(aspect_ratios_1_train, bins=10, alpha=0.5, label='Digit 1')
plt.xlabel('Aspect Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Aspect Ratios for Digits 0 and 1')
plt.legend()
plt.show()

######################################### 3 - Gaussian Distribution Bayesian Classification #########################################

# Calculating the prior probabilities based on the dataset
total_samples = len(filtered_labels)

# We already have caluculated half index, so we use that for the priors
prior_0 = 2 * half_index_0 / total_samples
prior_1 = 2 * half_index_1 / total_samples

# Print the priors
print("\nPrior Probability for Label 0:", prior_0)
print("Prior Probability for Label 1:", prior_1)

print("\n------------ Manual Implementation ------------")
# Calculate likelihoods for test images
# Given the training set 
PD0_is_D0 = norm.pdf(aspect_ratios_0_test, np.mean(aspect_ratios_0), np.std(aspect_ratios_0))
PD0_is_D1 = norm.pdf(aspect_ratios_0_test, np.mean(aspect_ratios_1), np.std(aspect_ratios_1))

PD1_is_D0 = norm.pdf(aspect_ratios_1_test, np.mean(aspect_ratios_0), np.std(aspect_ratios_0))
PD1_is_D1 = norm.pdf(aspect_ratios_1_test, np.mean(aspect_ratios_1), np.std(aspect_ratios_1))

BayesClass_D0 = ((PD0_is_D0 * prior_0) > (PD0_is_D1 * prior_1))
BayesClass_D1 = ((PD1_is_D1 * prior_1) > (PD1_is_D0 * prior_0))

########################################### 4 - Manual Prediction Accuracy Test ###########################################

count_errors_D0 = sum(BayesClass_D0 == False)
count_errors_D1 = sum(BayesClass_D1 == False)

# Total sum of errors
count_errors = count_errors_D0 + count_errors_D1

# Total number of test elements
test_el = len(aspect_ratios_0_test) + len(aspect_ratios_1_test)

# Print misclassified images and error percentage
print(f"\nMisclassified images: {count_errors_D0} (D0) + {count_errors_D1} (D1) = {count_errors}")
Error = (count_errors / test_el) * 100
print(f"\nError percentage: {Error:.2f}%")
print(f"Accuracy: {(100-Error):.2f}%\n")

########################################### 5 - Repeat with Bayesian Classifier ###########################################

print("\n------------ GaussianNB Implementation ------------")

from sklearn.naive_bayes import GaussianNB

# Prepare the training data, must reshape to use
X_train = np.concatenate((aspect_ratios_0, aspect_ratios_1)).reshape(-1, 1)

# ytrain has all the labels in the adjusted order
# instead of getting the actual labels we can put them like this because they are known
y_train = np.array([0]*len(aspect_ratios_0) + [1]*len(aspect_ratios_1))

# Prepare the test data
X_test = np.concatenate((aspect_ratios_0_test, aspect_ratios_1_test)).reshape(-1, 1)
y_test = np.array([0]*len(aspect_ratios_0_test) + [1]*len(aspect_ratios_1_test))

# Create and train the Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
# Predict labels for given test data:
y_pred = gnb.predict(X_test)

####################################### 6 - GaussianNB Prediction Accuracy Test #######################################

# Calculate the number of misclassified images
count_errors = (y_test != y_pred).sum()

Error = (count_errors / len(y_test)) * 100
Accuracy = 100 - Error

# Print results
print(f"\nMisclassified images: {count_errors_D0} (D0) + {count_errors_D1} (D1) = {count_errors}")
print(f"\nError percentage: {Error:.2f}%")
print(f"Accuracy: {Accuracy:.2f}%\n")


############################################# 7 - Visualize Results ##############################################

# Make a scatter plot that shows classification mistakes by index

pltTitle = "Classification of Digits 0 and 1 Using Gaussian Naive Bayes"
fig = plt.figure(figsize=(12, 6))
fig.canvas.manager.set_window_title(pltTitle)
plt.title(pltTitle)
plt.xlabel("Sample Index")
plt.ylabel("Aspect Ratio")

# Scatter plot of correctly classified samples
correctly_classified_indices = [i for i, correct in enumerate(y_test == y_pred) if correct]
correct_aspect_ratios = X_test[correctly_classified_indices]
plt.scatter(correctly_classified_indices, correct_aspect_ratios, color='green', marker='o', label='Correctly Classified')
# Scatter plot of incorrectly classified samples
all_indices = range(len(y_test))
# Get incorrectly classified indices as the complement of correctly classified indices
incorrectly_classified_indices = [i for i in all_indices if i not in correctly_classified_indices]
incorrect_aspect_ratios = X_test[incorrectly_classified_indices]
plt.scatter(incorrectly_classified_indices, incorrect_aspect_ratios, color='red', marker='x', label='Incorrectly Classified')
plt.legend()
plt.show()

# Result: 
# The result is logical because digit 1s have large aspect ratio variance, compared to digit 0s which generally have aspect ratio of 0.7 
# so when a digit 1 has aspect ratio of 0.7 or close it gets misclassified as a digit 0 as expected