import time
from collections import defaultdict
from config import *
from preprocessing import *

train_emails, train_labels = preprocess_dataset()

def spam_ham_separator(emails, labels):
    # Separate spam and ham emails
    spam_emails = {}
    ham_emails = {}
    for key, words in emails.items():
        if labels[key] == 1:
            spam_emails[key] = words
        else:
            ham_emails[key] = words
            
    return spam_emails, ham_emails

def vocab_count_generator(emails=train_emails, labels=train_labels):
    start_time = time.time()
    spam_emails, ham_emails = spam_ham_separator(emails, labels)
    spam_vocab = defaultdict(int)
    ham_vocab = defaultdict(int)
    
    for words in spam_emails.values():
        for word in words:
            spam_vocab[word] += 1
            
    for words in ham_emails.values():
        for word in words:
            ham_vocab[word] += 1

    # Prepare vocabulary and totals
    all_words = set(spam_vocab) | set(ham_vocab)
    vocab_size = len(all_words)
    spam_word_count = sum(spam_vocab.values())
    ham_word_count = sum(ham_vocab.values())
    
    return spam_vocab, ham_vocab, spam_word_count, ham_word_count, spam_emails, ham_emails

if __name__ == "__main__":
    vocab_count_generator()