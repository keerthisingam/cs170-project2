import numpy as np

def leave_one_out_cross_validation(data, feature_set, feature_to_add=None):
    selected_features = list(feature_set)  
    number_correctly_classified = 0
    n = data.shape[0]

    feature_matrix = data[:, selected_features]  

    for i in range(n):
        label_to_classify = data[i, 0]
        instance_features = feature_matrix[i]

        distances = np.sqrt(np.sum((feature_matrix - instance_features) ** 2, axis=1))  
        distances[i] = np.inf  

        nearest_neighbor_index = np.argmin(distances)  
        nearest_neighbor_label = data[nearest_neighbor_index, 0]  

        if label_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / n
    return accuracy

def forward_selection(data):
    num_features = data.shape[1] - 1  # Exclude label column
    current_set = []
    best_overall_accuracy = 0
    best_feature_set = []

    base_accuracy = leave_one_out_cross_validation(data, current_set, None)
    print(f"Base accuracy with no features: {base_accuracy:.4f}")

    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        best_feature = None
        best_accuracy = 0

        available_features = set(range(1, num_features + 1)) - set(current_set) #added for optimization

        for feature in available_features:
            accuracy = leave_one_out_cross_validation(data, current_set + [feature], feature)
            print(f"Testing feature set {current_set + [feature]}, Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature

        if best_feature:
            current_set.append(best_feature)
            print(f"Feature {best_feature} added, new set: {current_set}, Accuracy: {best_accuracy:.4f}")

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = list(current_set)
            else:
                print("\nWarning! Accuracy has decreased! Continuing search in case of local maxima")

    return best_feature_set, best_overall_accuracy

def backward_selection(data):
    """ Performs backward selection to find the best feature subset. Uses a stubbed cross-validation function (random accuracy). """
    
    num_features = data.shape[1] - 1  # Exclude label column
    current_set = list(range(1, num_features + 1))  #starts with all features
    best_overall_accuracy = 0
    best_feature_set = list(current_set)

    base_accuracy = leave_one_out_cross_validation(data, current_set,None)
    print(f"Base accuracy with all features: {base_accuracy:.4f}")

    for i in range(num_features - 1):
        print(f"\nOn level {i+1} of the search tree")
        worst_feature = None  #tracks the worst feature to remove in this iteration
        best_accuracy = 0  #stores the highest accuracy found in this iteration

        for feature in current_set:
            temp_set = [f for f in current_set if f != feature]  #removes one feature at a time
            accuracy = leave_one_out_cross_validation(data, temp_set,feature)  # Stub function
            print(f"Testing feature set {temp_set}, Accuracy: {accuracy:.4f}")  # Printing rn with random accuracy for every feature set

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature

        if worst_feature is not None:
            current_set.remove(worst_feature)
            print(f"Feature {worst_feature} removed, new set: {current_set}, Accuracy: {best_accuracy:.4f}")

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = list(current_set)
            else:
                print(f"\nWarning! Accuracy has decreased! Continuing search in case of local maxima")

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
        print("\n===== Running Forward Selection =====")
        best_features, accuracy = forward_selection(data)
        print(f"\nBest feature set (Forward Selection): {best_features}, Accuracy: {accuracy:.4f}")
    elif choice == "2":
        print("\n===== Running Backward Elimination =====")
        best_features, accuracy = backward_selection(data)
        print(f"\nBest feature set (Backward Elimination): {best_features}, Accuracy: {accuracy:.4f}")
    else:
        print("Invalid choice. Please restart and select either 1 or 2.")