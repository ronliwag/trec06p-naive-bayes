#counts the probabilities for prior and for each word, no smoothing is applied yet and 
#crossmatching of unique words for spam and ham is not applied yet

from config import *
from preprocessing import *
import time


def count_statistics():
    start_time = time.time()
    filtered_email, labels = read_and_print_dataset()

    #separate spam and ham from filtered_email
    spam_emails = {}
    ham_emails = {}
    for key in filtered_email:
        if labels[key] == 'spam':
            spam_emails.update({key : filtered_email[key]})
        else:
            ham_emails.update({key : filtered_email[key]})

    #count word instances for each spam/ham
    spam_vocab = {}
    ham_vocab = {}
    spam_word_count = 0
    ham_word_count = 0
    for key in spam_emails:
        for word in spam_emails[key]:
            spam_word_count += 1
            if word not in spam_vocab.keys():
                spam_vocab.update({word : 1})
            else:
                spam_vocab[word] += 1
    for key in ham_emails:
        for word in ham_emails[key]:
            ham_word_count += 1
            if word not in ham_vocab.keys():
                ham_vocab.update({word : 1})
            else:
                ham_vocab[word] += 1
    
    total_word_count = spam_word_count + ham_word_count
    spam_prior = spam_word_count / total_word_count
    ham_prior = ham_word_count / total_word_count
    print('\n')
    print(f"n(spam) = {spam_word_count}")
    print(f"n(ham) = {ham_word_count}")
    print(f"n(total) = {total_word_count}")
    print(f"P(spam) = {round(spam_prior, 6)}")
    print(f"P(hpam) = {round(ham_prior, 6)}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    print('\n')
    
    #get word probabilities
    spam_vocab_probability, ham_vocab_probability = word_probability(spam_vocab, ham_vocab, spam_word_count, ham_word_count)
    
    #for word in spam_vocab_probability:
    #    print(f"{word} = {spam_vocab_probability[word]}")
    #print('\n\n')
    #for word in ham_vocab_probability:
    #    print(f"{word} = {ham_vocab_probability[word]}")
    
def word_probability(spam_vocab, ham_vocab, spam_word_count, ham_word_count):
    spam_vocab_probability = {}
    ham_vocab_probability = {}
    
    for key in spam_vocab:
        probability = spam_vocab[key]/spam_word_count
        spam_vocab_probability.update({key : round(probability, 6)})
        
    for key in ham_vocab:
        probability = ham_vocab[key]/ham_word_count
        ham_vocab_probability.update({key : round(probability, 6)})
        
    return spam_vocab_probability, ham_vocab_probability

if __name__ == "__main__":
    count_statistics()