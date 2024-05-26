import numpy as np
import pandas as pd
from collections import defaultdict
import random
import queue
import copy


class Node:
    def __init__(self, state: set, parent, accuracy):
        self.state = state
        self.parent = parent
        self.accuracy = accuracy

    def printstate(self):
        print(self.state)

    def evaluate(self):
        self.accuracy = random.uniform(0, 1)

def expand_node(node, total_features):
    children = []
    current_features = set(node.state)
    for feature in range(1, total_features + 1):
        if feature not in current_features:
            new_state = set(node.state)
            new_state.add(feature)
            child_node = Node(state = new_state, parent = node, accuracy = 0)
            child_node.evaluate()
            children.append(child_node)
    return children

def expand_node_be(node, total_features):
    children = []
    current_features = set(node.state)
    for feature in range(1, total_features + 1):
        if feature in current_features:
            new_state = set(node.state)
            new_state.remove(feature)
            child_node = Node(state = new_state, parent = node, accuracy = 0)
            child_node.evaluate()
            children.append(child_node)
    return children

def forward_selection(n: int):
    root = Node(state = {}, parent = None, accuracy = 0)
    root.evaluate()
    best_node = root
    currnode = root
    goalstate = {i for i in range(1, n + 1)}

    print(f"Using no features and \"random\" evaluation, I get an accuracy of {round(root.accuracy * 100, 2)}%\n")
    print("Beginning search\n")
    while currnode.state != goalstate:
        children = expand_node(currnode, n)
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

def backward_elimination(n):
    startstate = {i for i in range(1, n + 1)}
    root = Node(state = startstate, parent = None, accuracy = 0)
    root.evaluate()
    best_node = root
    currnode = root
    print(f"Using all features and feature set {root.state} and \"random\" evaluation, I get an accuracy of {round(root.accuracy * 100, 2)}%\n")
    print("Beginning search\n")
    while currnode.state != set({}):
        children = expand_node_be(currnode, n)
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

def best_n_features():
    pass



def main():
    print("Welcome to our groups Feature Selection Algorithm.")
    num_ftrs = int(input("Please enter number of features: "))


    print("Type the number of the algorithm you want to run:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. k-Best Features")

    option = int(input())
    
    if option == 1:    
        best = forward_selection(num_ftrs)
    elif option == 2:
        best = backward_elimination(num_ftrs)
    else:
        print("Method not implemented yet!")

if __name__ == "__main__":
    main()
