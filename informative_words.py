import math
from classifier import *
from preprocessing import *
from count_statistics import *
from evaluation import compute_precision_recall

def calculate_word_informativeness():
    """Calculate informativeness of words using mutual information."""
    
    # Get the vocabulary and counts
    spam_vocab, ham_vocab, spam_word_count, ham_word_count, spam_emails, ham_emails = vocab_count_generator()
    
    # Get total documents
    total_docs = len(spam_emails) + len(ham_emails)
    spam_docs = len(spam_emails)
    ham_docs = len(ham_emails)
    
    # Prior probabilities
    p_spam = spam_docs / total_docs
    p_ham = ham_docs / total_docs
    
    # Combine vocabularies
    all_words = set(spam_vocab.keys()).union(set(ham_vocab.keys()))
    
    # Calculate mutual information for each word
    word_scores = {}
    
    for word in all_words:
        # Count documents containing the word
        spam_word_docs = sum(1 for docs in spam_emails.values() if word in docs)
        ham_word_docs = sum(1 for docs in ham_emails.values() if word in docs)
        
        # Skip words that appear in very few documents
        if spam_word_docs + ham_word_docs < 5:
            continue
            
        # Calculate probabilities
        p_word = (spam_word_docs + ham_word_docs) / total_docs
        p_word_given_spam = spam_word_docs / spam_docs if spam_docs > 0 else 0
        p_word_given_ham = ham_word_docs / ham_docs if ham_docs > 0 else 0
        
        # Calculate mutual information components
        mi_spam = 0
        mi_ham = 0
        
        # Word appears in document
        if p_word_given_spam > 0:
            mi_spam = p_spam * p_word_given_spam * math.log2(p_word_given_spam / p_word)
        
        if p_word_given_ham > 0:
            mi_ham = p_ham * p_word_given_ham * math.log2(p_word_given_ham / p_word)
        
        # Word doesn't appear in document
        p_not_word_given_spam = 1 - p_word_given_spam
        p_not_word_given_ham = 1 - p_word_given_ham
        p_not_word = 1 - p_word
        
        if p_not_word_given_spam > 0 and p_not_word > 0:
            mi_spam += p_spam * p_not_word_given_spam * math.log2(p_not_word_given_spam / p_not_word)
        
        if p_not_word_given_ham > 0 and p_not_word > 0:
            mi_ham += p_ham * p_not_word_given_ham * math.log2(p_not_word_given_ham / p_not_word)
        
        # Total mutual information for this word
        mi_total = mi_spam + mi_ham
        
        # Store the score and whether it's more indicative of spam or ham
        is_spam_indicator = p_word_given_spam > p_word_given_ham
        word_scores[word] = (mi_total, is_spam_indicator)
    
    return word_scores

def get_top_informative_words(n=200):
    """Get the top n most informative words, separated for spam and ham."""
    word_scores = calculate_word_informativeness()
    
    # Sort words by informativeness score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    # Separate spam and ham indicators
    spam_indicators = [(word, score) for word, (score, is_spam) in sorted_words if is_spam]
    ham_indicators = [(word, score) for word, (score, is_spam) in sorted_words if not is_spam]
    
    # Get top words for each category
    top_spam = spam_indicators[:n]
    top_ham = ham_indicators[:n]
    
    # Return combined list of top words
    top_words = set([word for word, _ in (top_spam + top_ham)[:n]])
    
    return top_words, top_spam, top_ham

def classify_with_top_words(message, top_words, lambda_value):
    """Classify a message using only the top informative words."""
    # Filter message to only include top words
    filtered_message = [word for word in message if word in top_words]
    
    # If no top words are in the message, return the original message
    if not filtered_message:
        return classify_message(message, lambda_value)
    
    # Classify using only the top words
    return classify_message(filtered_message, lambda_value)

def evaluate_with_top_words(lambda_value=0.1, n=200):
    """Evaluate the classifier using only the top n informative words."""
    print(f"\n=== Evaluating with top {n} informative words, Lambda: {lambda_value} ===")
    
    # Get top words
    top_words, top_spam_words, top_ham_words = get_top_informative_words(n)
    
    # Print top words for spam and ham
    print(f"\nTop 10 Spam Indicator Words:")
    for word, score in top_spam_words[:10]:
        print(f"  {word}: {score:.6f}")
    
    print(f"\nTop 10 Ham Indicator Words:")
    for word, score in top_ham_words[:10]:
        print(f"  {word}: {score:.6f}")
    
    # Load test data
    test_emails, test_labels = preprocess_dataset(dir_index=DIR_INDEX)
    
    # Set up counters
    true_positives = 0 
    false_positives = 0  
    true_negatives = 0  
    false_negatives = 0
    
    # Classify using only top words
    for key in test_emails.keys():
        actual_label = test_labels[key]
        words = test_emails[key]
        
        # Filter message to only include top words
        filtered_words = [word for word in words if word in top_words]
        
        # If no top words found, use original message
        if not filtered_words:
            predicted_label = classify_message(words, lambda_value)
        else:
            predicted_label = classify_message(filtered_words, lambda_value)
        
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
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"\n=== Evaluation Results with Top {n} Words ===")
    print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    return top_words, precision, recall, f1_score

if __name__ == "__main__":
    # Find the best lambda from previous runs (you would replace this with your best lambda)
    best_lambda = 0.1
    
    # Evaluate using top 200 words
    top_words, precision, recall, f1 = evaluate_with_top_words(lambda_value=best_lambda, n=200)
    
    # Save top words to file for reference
    with open("top_200_words.txt", "w") as f:
        for word in sorted(top_words):
            f.write(f"{word}\n")