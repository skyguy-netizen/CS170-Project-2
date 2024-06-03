import pandas as pd
from algorithms import read_file, FeatureSearch
from validator import Validator
import time

def main():
    print("Welcome to our groups Feature Selection plus Nearest Neighbor model.")
    filename = input("Please enter name of file: ")
    print(filename)

    option = input("Would you like to use a feature selection algorithm or use your subset of features? (yes or no): ")
    print(option)

    k = int(input("Enter value for k (nearest-neighbors): "))
    print(k)

    if option == "yes":
        print("Type the number of the feature selection algorithm you want to run:")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        print("3. k-Best Features")
        choice = int(input())
        data = read_file(filename)
        search_object = FeatureSearch(data, k)
        if choice == 1:
            print("Running forward selection here")
            best = search_object.forward_selection()
        elif choice == 2:
            print("Running backward elimination here")
            best = search_object.backward_elimination()
        elif choice == 3:
            print("Method not implemented yet!")
            best = search_object.k_best_features()
    else:
        ftrs_input =  input("Please enter feature numbers seperated by space: ")
        print(ftrs_input)
        features = list(map(int,ftrs_input.split(' ')))
        print("Reading dataset")
        start_time = time.time()
        data = read_file(filename)
        run_time = time.time() - start_time
        print(f"Finished reading dataset; time taken is {run_time} seconds ")
        print()
        print("Running leave one out validation with features")
        start_time = time.time()
        evalu = Validator(data[features +  ['class']])
        accuracy = evalu.leave_one_out()
        run_time = time.time() - start_time
        print(f"Finished evaluating model; time taken is {run_time} seconds ")
        print()
        print(f"Accuracy with feature set {set(features)} is {accuracy * 100}%")

if __name__ == "__main__":
    main()
