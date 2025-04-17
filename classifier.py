import math
import re
from count_statistics import count_statistics  # Import the function, not individual variables

# Call the function to get the values needed for classification
spam_vocab, ham_vocab, spam_word_count, ham_word_count, spam_prior, ham_prior = count_statistics()

# Combine both vocabularies to get the overall vocabulary for Laplace smoothing
all_words = set(spam_vocab.keys()).union(set(ham_vocab.keys()))
vocab_size = len(all_words)

def tokenize(message):
    """
    Tokenize the message using alphabetic sequences, as per instructions.
    """
    return re.findall(r'\b[a-zA-Z]+\b', message.lower())

def classify_message(message):
    """
    Classify a single message as 'spam' or 'ham' using Naive Bayes.
    """
    words = tokenize(message)

    # Start with the log of prior probabilities
    log_spam = math.log(spam_prior) if spam_prior > 0 else float('-inf')
    log_ham = math.log(ham_prior) if ham_prior > 0 else float('-inf')

    for word in words:
        # Laplace smoothing
        spam_word_freq = spam_vocab.get(word, 0)
        ham_word_freq = ham_vocab.get(word, 0)

        spam_prob = (spam_word_freq + 1) / (spam_word_count + vocab_size)
        ham_prob = (ham_word_freq + 1) / (ham_word_count + vocab_size)

        log_spam += math.log(spam_prob)
        log_ham += math.log(ham_prob)

    return 'spam' if log_spam > log_ham else 'ham'

def classify_test_messages(test_messages):
    """
    Test function: classify multiple messages and print results.
    """
    for idx, msg in enumerate(test_messages, 1):
        prediction = classify_message(msg)
        print(f"Message {idx}: '{msg}' → Classified as: {prediction}")

# Example usage (replace these messages with your test set content)
if __name__ == "__main__":
    test_samples = [
        "Congratulations! You have won a free vacation!",
        "Hi, don't forget our lunch meeting today.",
        "Limited time offer! Claim your prize now!",
        "Can you send me the notes from yesterday’s class?",
    ]

    classify_test_messages(test_samples)
