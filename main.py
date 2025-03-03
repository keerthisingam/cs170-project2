import numpy as np

def leave_one_out_cross_validation(data, feature_set):
    "Stub function: This is a placeholder for cross-validation. Initially, it does not compute accuracy and returns a random number."
    return np.random.rand()  # Placeholder accuracy (random value)

def forward_selection(data):
    " Performs forward selection to find the best feature subset. Uses a stubbed cross-validation function (random accuracy)."

    num_features = data.shape[1] - 1  # Exclude label column
    current_set = []
    best_overall_accuracy = 0
    best_feature_set = []

    for i in range(num_features):
        best_feature = None #tracks the best feature in this iteration
        best_accuracy = 0 #stores the highest accuracy found in this iteration

        for feature in range(1, num_features + 1): #looping through all features not including already selected ones
            if feature not in current_set:
                temp_set = current_set + [feature]
                accuracy = leave_one_out_cross_validation(data, temp_set)  # Stub function
                print(f"Testing feature set {temp_set}, Accuracy: {accuracy:.4f}") #printing rn with random accuracy for every feature set

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

        if best_feature:
            current_set.append(best_feature)
            print(f"Feature {best_feature} added, new set: {current_set}, Accuracy: {best_accuracy:.4f}")

            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = list(current_set)

    return best_feature_set, best_overall_accuracy


if __name__ == "__main__":
    # Load dataset (assuming first column is class labels, rest are features)
    data = np.loadtxt('CS170_Small_Data__21.txt')

    print("\n===== Forward Selection (Stub Version) =====")
    best_forward_features, forward_accuracy = forward_selection(data)
    print(f"Best feature set (Forward Selection): {best_forward_features}, Accuracy: {forward_accuracy:.4f}")
