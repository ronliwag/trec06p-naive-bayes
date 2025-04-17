import math
import re
from count_statistics import * # Import the function, not individual variables
from preprocessing import extract_words

# Call the function to get the values needed for classification
spam_vocab, ham_vocab, spam_word_count, ham_word_count, spam_emails, ham_emails = vocab_count_generator()

def train_from_data(spam_emails, ham_emails):
    total_emails = len(spam_emails) + len(ham_emails)
    spam_prior = len(spam_emails) / total_emails
    ham_prior = len(ham_emails) / total_emails
    
    return spam_prior, ham_prior


def classify_message(message, lambda_value):
    spam_prior, ham_prior = train_from_data(spam_emails, ham_emails)
    
    # Combine both vocabularies to get the overall vocabulary for Laplace smoothing
    all_words = set(spam_vocab.keys()).union(set(ham_vocab.keys()))
    vocab_size = len(all_words)
    
    # Start with the log of prior probabilities
    log_spam = math.log(spam_prior) if spam_prior > 0 else float('-inf')
    log_ham = math.log(ham_prior) if ham_prior > 0 else float('-inf')

    for word in message:
        # Laplace smoothing (add lambda to numerator and lambda*vocab_size to denominator)
        spam_word_freq = spam_vocab.get(word, 0)
        ham_word_freq = ham_vocab.get(word, 0)

        # Calculate probabilities with consistent smoothing
        spam_prob = (spam_word_freq + lambda_value) / (spam_word_count + lambda_value * vocab_size)
        ham_prob = (ham_word_freq + lambda_value) / (ham_word_count + lambda_value * vocab_size)

        # Safely accumulate log probabilities
        log_spam += math.log(spam_prob) if spam_prob > 0 else float('-inf')
        log_ham += math.log(ham_prob) if ham_prob > 0 else float('-inf')

    return 1 if log_spam > log_ham else 0

def classify_test_messages(test_messages, lambda_value=1.0):
    test_classification = {}
    idx = 0
    for msg in test_messages:
        test_classification.update({idx : extract_words(msg)})
        idx += 1
        
    for idx, msg in test_classification.items():
        prediction = classify_message(msg, lambda_value)
        print(f"Message {idx}: '{test_messages[idx]}' → Classified as: {prediction}")

# Example usage (replace these messages with your test set content)
if __name__ == "__main__":
    test_samples = [
        "Congratulations! You have won a free vacation!",
        "Hi, don't forget our lunch meeting today.",
        "Limited time offer! Claim your prize now!",
        "Can you send me the notes from yesterday’s class?",
    ]

    classify_test_messages(test_samples)