import numpy as np
import time

def leave_one_out_cross_validation(data, feature_set, feature_to_add=None):
    selected_features = list(feature_set)  
    number_correctly_classified = 0
    n = data.shape[0]

    feature_matrix = data[:, selected_features]  #extracts selected feature columns

    for i in range(n):
        label_to_classify = data[i, 0]
        instance_features = feature_matrix[i]

        #computes Euclidean distances from the current sample to all others
        distances = np.sqrt(np.sum((feature_matrix - instance_features) ** 2, axis=1))  
        distances[i] = np.inf  

        nearest_neighbor_index = np.argmin(distances)  
        nearest_neighbor_label = data[nearest_neighbor_index, 0]  #gets the label of the nearest neighbor

        if label_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / n
    return accuracy

def forward_selection(data):
    num_features = data.shape[1] - 1  #excludes label column
    current_set = []
    best_overall_accuracy = 0
    best_feature_set = []


    base_accuracy = leave_one_out_cross_validation(data, current_set, None)
    print(f"Base accuracy with no features: {base_accuracy * 100:.1f}%")
    best_overall_accuracy = base_accuracy

    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        best_feature = None
        best_accuracy = 0

        for feature in range(1, num_features + 1): #looping through all features not including already selected ones
            if feature not in current_set:
                accuracy = leave_one_out_cross_validation(data, current_set + [feature],feature) 
                print(f"  Using feature(s) {current_set + [feature]} accuracy is {accuracy * 100:.1f}%") 

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature


        if best_feature:
            current_set.append(best_feature)
            print(f"Feature {best_feature} added, current best feature set: {current_set} accuracy is {best_accuracy * 100:.1f}%")

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = list(current_set)
            else:
                print("\nWarning! Accuracy has decreased! Continuing search in case of local maxima")

    return best_feature_set, best_overall_accuracy

def backward_selection(data):
    """ Performs backward selection to find the best feature subset. Uses a stubbed cross-validation function (random accuracy). """
    
    num_features = data.shape[1] - 1  #exclude label column
    current_set = list(range(1, num_features + 1))  #starts with all features
    best_overall_accuracy = 0
    best_feature_set = list(current_set)

    base_accuracy = leave_one_out_cross_validation(data, current_set,None)
    print(f"Base accuracy with all features: {base_accuracy * 100:.1f}%")
    best_overall_accuracy = base_accuracy

    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        worst_feature = None  #tracks the worst feature to remove in this iteration
        best_accuracy = 0  #stores the highest accuracy found in this iteration

        for feature in current_set:
            temp_set = [f for f in current_set if f != feature]  #removes one feature at a time
            accuracy = leave_one_out_cross_validation(data, temp_set,feature) 
            print(f"  Using feature(s) {temp_set} accuracy is {accuracy * 100:.1f}%") 

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature

        if worst_feature is not None:
            current_set.remove(worst_feature)
            print(f"Feature {worst_feature} removed, current best feature set: {current_set} accuracy is {best_accuracy * 100:.1f}%")

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = list(current_set)
            else:
                print(f"\nWarning! Accuracy has decreased! Continuing search in case of local maxima")

    #evaluates the empty set after the loop
    empty_set_accuracy = leave_one_out_cross_validation(data, [], None)
    print(f"  Using feature(s) {{}} accuracy is {empty_set_accuracy * 100:.1f}%")

    if empty_set_accuracy > best_overall_accuracy:
        best_overall_accuracy = empty_set_accuracy
        best_feature_set = []

    return best_feature_set, best_overall_accuracy

if __name__ == "__main__":
    print("Welcome to Keerthi Singam's Feature Selection Algorithm.")

    # Prompt user for file name
    file_name = input("Type in the name of the file to test: ")
    
    try:
        data = np.loadtxt(file_name)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    # Prompt user for algorithm choice
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input("\nEnter your choice (1 or 2): ")

    if choice == "1":
        print("\nBeginning search.")
        start_time = time.time()  #start timer
        best_features, accuracy = forward_selection(data)
        end_time = time.time()  #end timer

        elapsed_time = end_time - start_time  #calculates elapsed time
        print(f"\nBest feature set: {best_features}, Accuracy: {accuracy * 100:.1f}%")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    elif choice == "2":
        print("\nBeginning search.")
        start_time = time.time()  #start timer
        best_features, accuracy = backward_selection(data)
        end_time = time.time()  #end timer

        elapsed_time = end_time - start_time  #calculate elapsed time
        print(f"\nBest feature set: {best_features}, Accuracy: {accuracy * 100:.1f}%")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    else:
        print("Invalid choice. Please restart and select either 1 or 2.")