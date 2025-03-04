import numpy as np

def leave_one_out_cross_validation(data,feature_set,feature_to_add=None):

    selected_features = list(feature_set)

    number_correctly_classified = 0
    n = data.shape[0]

    for i in range(n):
        label_to_classify = data[i, 0]
        instance_features = data[i, selected_features]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None


        for j in range(n):
            if j != i:
                neighbor_features = data[j, selected_features]
                distance = np.sqrt(np.sum((instance_features - neighbor_features)**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = data[j, 0]


        if label_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / n
    return accuracy
    
def forward_selection(data):
    " Performs forward selection to find the best feature subset. Uses a stubbed cross-validation function (random accuracy)."

    num_features = data.shape[1] - 1  # Exclude label column
    current_set = []
    best_overall_accuracy = 0
    best_feature_set = []

    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        best_feature = None #tracks the best feature in this iteration
        best_accuracy = 0 #stores the highest accuracy found in this iteration

        for feature in range(1, num_features + 1): #looping through all features not including already selected ones
            if feature not in current_set:
                # temp_set = current_set + [feature]
                accuracy = leave_one_out_cross_validation(data, current_set + [feature],feature)  # Stub function
                print(f"Testing feature set {current_set + [feature]}, Accuracy: {accuracy:.4f}") #printing rn with random accuracy for every feature set

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
                print(f"\nWarning! Accuracy has decreased! Continuing search in case of local maxima")

    return best_feature_set, best_overall_accuracy

def backward_selection(data):
    """ Performs backward selection to find the best feature subset. Uses a stubbed cross-validation function (random accuracy). """
    
    num_features = data.shape[1] - 1  # Exclude label column
    current_set = list(range(1, num_features + 1))  # Start with all features
    best_overall_accuracy = 0
    best_feature_set = list(current_set)

    for i in range(num_features - 1):
        print(f"\nOn level {i+1} of the search tree")
        worst_feature = None  # Tracks the worst feature to remove in this iteration
        best_accuracy = 0  # Stores the highest accuracy found in this iteration

        for feature in current_set:
            temp_set = [f for f in current_set if f != feature]  # Remove one feature at a time
            
            accuracy = leave_one_out_cross_validation(data, temp_set,feature)  # Stub function
            print(temp_set)
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
    # Load dataset (assuming first column is class labels, rest are features)
    data = np.loadtxt('CS170_Large_Data__118.txt')

    print("\n===== Forward Selection (Stub Version) =====")
    best_forward_features, forward_accuracy = forward_selection(data)
    print(f"Best feature set (Forward Selection): {best_forward_features}, Accuracy: {forward_accuracy:.4f}")

    print("\n===== Backward Selection (Stub Version) =====")
    best_backward_features, backward_accuracy = backward_selection(data)
    print(f"Best feature set (Backward Selection): {best_backward_features}, Accuracy: {backward_accuracy:.4f}")
