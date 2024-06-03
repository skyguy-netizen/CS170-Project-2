import random
import queue
import copy
from collections import defaultdict
import numpy as np
import pandas as pd
from validator import Validator

class Node:
    def __init__(self, state: set, parent, accuracy):
        self.state = state
        self.parent = parent
        self.accuracy = accuracy

    def printstate(self):
        print(self.state)

    def evaluate(self, data, k = 1):
        evalu = Validator(data[list(self.state) + ['class']], k)
        self.accuracy = evalu.leave_one_out()

class FeatureSearch:
    def __init__(self, data, k = 1):
        self.data = data
        self.total_features = len(self.data.columns) - 1
        self.k = k

    def expand_node_fs(self, node):
        children = []
        current_features = set(node.state)
        for feature in range(1, self.total_features + 1):
            if feature not in current_features:
                new_state = set(node.state)
                new_state.add(feature)
                child_node = Node(state = new_state, parent = node, accuracy = 0)
                child_node.evaluate(self.data, self.k)
                children.append(child_node)
        return children

    def expand_node_be(self, node):
        children = []
        current_features = set(node.state)
        for feature in range(1, self.total_features + 1):
            if feature in current_features:
                new_state = set(node.state)
                new_state.remove(feature)
                child_node = Node(state = new_state, parent = node, accuracy = 0)
                child_node.evaluate(self.data, self.k)
                children.append(child_node)
        return children

    def forward_selection(self):
        root = Node(state = {}, parent = None, accuracy = 0)
        root.evaluate(self.data)
        best_node = root
        currnode = root
        goalstate = {i for i in range(1, self.total_features + 1)}

        print(f"Using no features and leave-one-out evaluation, I get an accuracy of {round(root.accuracy * 100, 2)}%\n")
        print("Beginning search\n")
        while currnode.state != goalstate:
            children = self.expand_node_fs(currnode)
            curr_best_node = children[0]
            for child in children:
                print(f"\tUsing feature(s) {child.state} accuracy is {round(child.accuracy * 100, 2)}%")
                if child.accuracy > curr_best_node.accuracy:
                    curr_best_node = child
            currnode = curr_best_node
            print(f"\nFeature set {currnode.state} was best, accuracy is {round(currnode.accuracy * 100, 2)}%\n")
            if currnode.accuracy > best_node.accuracy:
                best_node = currnode
            else:
                print("Warning: Accuracy has decreased, still continuing search\n")
        print(f"Finished search!!! The best feature set is {best_node.state} with an accuracy of {round(best_node.accuracy * 100, 2)}%")

    def backward_elimination(self,):
        startstate = {i for i in range(1, self.total_features + 1)}
        root = Node(state = startstate, parent = None, accuracy = 0)
        root.evaluate(self.data)
        best_node = root
        currnode = root
        print(f"Using all features and feature set {root.state} and leave-one-out evaluation, I get an accuracy of {round(root.accuracy * 100, 2)}%\n")
        print("Beginning search\n")
        while currnode.state != set({}):
            children = self.expand_node_be(currnode)
            curr_best_node = children[0]
            for child in children:
                print(f"\tUsing feature(s) {child.state} accuracy is {round(child.accuracy * 100, 2)}%")
                if child.accuracy > curr_best_node.accuracy:
                    curr_best_node = child
            currnode = curr_best_node
            print(f"\nFeature set {currnode.state} was best, accuracy is {round(currnode.accuracy * 100, 2)}%\n")
            if currnode.accuracy > best_node.accuracy:
                best_node = currnode
            else:
                print("Warning: Accuracy has decreased, still continuing search\n")
        print(f"Finished search!!! The best feature set is {best_node.state} with an accuracy of {round(best_node.accuracy * 100, 2)}%")
    def k_best_features(self, k):
        feature_accuracies = []
        for feature in range(1, self.total_features + 1):
            node = Node(state = {feature}, parent = None, accuracy = 0)
            node.evaluate(self.data, self.k)
            feature_accuracies.append((feature, node.accuracy))

        # Sort features by accuracy in descending order
        feature_accuracies.sort(key=lambda x: x[1], reverse=True)

        # Select the top k features
        best_features = set([feature for feature, accuracy in feature_accuracies[:k]])

        # Evaluate the k best features as a set
        best_node = Node(state = best_features, parent = None, accuracy = 0)
        best_node.evaluate(self.data, self.k)

        print(f"Selected top {k} features: {best_features} with accuracy: {round(best_node.accuracy * 100, 2)}%")
def read_file(filename):
    try:
        file = open(filename, "r")
        lines = file.readlines()
    except:
        print("File not found!")
        exit(1)

    data = [line.split() for line in lines]
    data = np.array(data).astype(float)

    classes = data[:, 0]
    features = data[:, 1:]

    data_dict = {(i+1): features[:, i] for i in range(features.shape[1])}
    data_dict['class'] = classes
    
    return pd.DataFrame(data_dict)