import time
import math
from classifier import *
from preprocessing import *

def compute_precision_recall(l_val):
    true_positives = 0 
    false_positives = 0  
    true_negatives = 0  
    false_negatives = 0
    
    test_keys = test_emails.keys()
    for key in test_keys:
        actual_label = test_labels[key]
        words = test_emails[key]
        predicted_label = classify_message(words, lambda_value=l_val)
        
    # Update counters
        if predicted_label == 1 and actual_label == 1:
            true_positives += 1
        elif predicted_label == 1 and actual_label == 0:
            false_positives += 1
        elif predicted_label == 0 and actual_label == 0:
            true_negatives += 1
        elif predicted_label == 0 and actual_label == 1:
            false_negatives += 1
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    # Avoid division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"\n=== Evaluation Metrics, Lambda: {l_val}===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


def lambda_precision_recall():
    start_time = time.time()
    lambda_values = [2.0, 1.0, 0.5, 0.1, 0.005]
    
    for l_val in lambda_values:
        compute_precision_recall(l_val)
    elapsed_time = time.time() - start_time
    print(f"Function took {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    test_emails, test_labels = preprocess_dataset(dir_index=DIR_INDEX)
    #compute_precision_recall()
    lambda_precision_recall()
