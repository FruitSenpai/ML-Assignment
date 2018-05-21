# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:02:52 2018

@author: MO70
"""

import pandas
import math
import numpy
import sys
from random import seed
from random import randrange

# For Incresing the amount of Recursion Allowed
#sys.setrecursionlimit(5000)

adultdata = pandas.read_csv("training2.csv",  na_values=['?'])

for name in ["workclass","education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income"]:
   col = pandas.Categorical(adultdata[name])
   adultdata[name] = col.codes
#print adultdata.info()
   
AdultDataset = adultdata.values.tolist()   

# Split the dataset into the K_folds you want
def cross_validation_split(dataset, no_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / no_folds)
	for i in range(no_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate the percentage of the Accuracy Achieved
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Using a validation split we can evaluate the algorithm
def evaluate_algorithm(dataset, algorithm, no_folds, *args):
	folds = cross_validation_split(dataset, no_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Dataset is split that is based on an attribute and its value 
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Gini index for a split dataset is Calculated
def gini_index(groups, classes):
	
	no_instances = float(sum([len(start) for start in groups]))
	gini = 0.0
	for start in groups:
		size = float(len(start))
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in start].count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / no_instances)
	return gini

# Get the best split point from your dataset
def get_split(dataset, no_features):
	class_values = list(set(row[-1] for row in dataset))
	best_index, best_value, best_score, best_groups = 999, 999, 999, None
	features = list()
	while len(features) < no_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < best_score:
				best_index, best_value, best_score, best_groups = index, row[index], gini, groups
	return {'index':best_index, 'value':best_value, 'groups':best_groups}

# Create the initial terminal node
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Make the splits for the node or make it the terminial
def split(node, max_depth, min_size, no_features, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, no_features)
		split(node['left'], max_depth, min_size, no_features, depth+1)
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, no_features)
		split(node['right'], max_depth, min_size, no_features, depth+1)

# Make a decision tree for the algorithm
def build_tree(train, max_depth, min_size, no_features):
	root = get_split(train, no_features)
	split(root, max_depth, min_size, no_features, 1)
	return root

# With the Decision Tree we make a prediction
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a Sub-Dataset set from the Dataset for Training
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# After obtaining the Bagged Trees we can Make a Prediction
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# The Algorithm for the Random Forest
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, no_features):
	trees = list()
	for i in range(n_trees):
		mainsample = subsample(train, sample_size)
		tree = build_tree(mainsample, max_depth, min_size, no_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

# Run the Algorithm and Evaluate it
no_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
no_features = 8 
for no_trees in [1, 5, 10]:
	scores = evaluate_algorithm(AdultDataset, random_forest, no_folds, max_depth, min_size, sample_size, no_trees, no_features)
	print('Trees: %d' % no_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

  