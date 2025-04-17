import time
from collections import defaultdict
from config import *
from preprocessing import *

def count_statistics():
    start_time = time.time()
    
    # Load and preprocess data
    filtered_email, labels = read_and_print_dataset()
    
    # Separate spam and ham emails
    spam_emails = {}
    ham_emails = {}
    for key, words in filtered_email.items():
        if labels[key] == 'spam':
            spam_emails[key] = words
        else:
            ham_emails[key] = words

    # Count word frequencies with defaultdict
    spam_vocab = defaultdict(int)
    ham_vocab = defaultdict(int)
    
    for words in spam_emails.values():
        for word in words:
            spam_vocab[word] += 1
            
    for words in ham_emails.values():
        for word in words:
            ham_vocab[word] += 1

    # Calculate PRIORS (email counts, not word counts)
    total_emails = len(spam_emails) + len(ham_emails)
    spam_prior = len(spam_emails) / total_emails
    ham_prior = len(ham_emails) / total_emails

    # Prepare vocabulary and totals
    all_words = set(spam_vocab) | set(ham_vocab)
    vocab_size = len(all_words)
    spam_word_count = sum(spam_vocab.values())
    ham_word_count = sum(ham_vocab.values())

    # Calculate word probabilities WITH LAPLACE SMOOTHING
    spam_probs = {}
    ham_probs = {}
    alpha = 1  # Smoothing factor
    
    for word in all_words:
        spam_probs[word] = (spam_vocab.get(word, 0) + alpha) / (spam_word_count + alpha * vocab_size)
        ham_probs[word] = (ham_vocab.get(word, 0) + alpha) / (ham_word_count + alpha * vocab_size)

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Spam emails: {len(spam_emails)}")
    print(f"Ham emails: {len(ham_emails)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"P(spam): {spam_prior:.6f}")
    print(f"P(ham): {ham_prior:.6f}")
    print(f"Execution time: {time.time() - start_time:.2f}s\n")
    
    return spam_prior, ham_prior, spam_probs, ham_probs

if __name__ == "__main__":
    spam_prior, ham_prior, spam_probs, ham_probs = count_statistics()