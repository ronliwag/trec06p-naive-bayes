#cleans each email down to the tokenized unique words for each email body stored in an array

import os
import nltk
from nltk.corpus import stopwords
from config import *
from collections import defaultdict
from email import message_from_string


def extract_email_body(raw_email):
    try:
        msg = message_from_string(raw_email)
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                    
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode('latin-1', errors='replace')
                    break
        else:
            body = msg.get_payload(decode=True).decode('latin-1', errors='replace')
            
        return body if body.strip() else raw_email  # Fallback to raw if empty
    except Exception as e:
        print(f"Email parsing error: {str(e)}")
        return raw_email  # Fallback to original text

def read_and_print_dataset(): #remove parameter to remove limit
    # Paths
    email_dir = DIR_DATA
    label_path = DIR_INDEX
    
    # Check if paths exist
    if not os.path.exists(email_dir):
        print(f"Error: Email directory not found at {email_dir}")
        return
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return
    
    # Read labels
    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label, email_file = line.split()
                labels[email_file] = label
            except ValueError:
                print(f"Skipping malformed line: {line}")
    
    # Read and print sample emails
    email_count = 0
    email_strip = {}
    for email_file in labels.keys():
        email_path = os.path.join(email_dir, email_file)
        try:
            with open(email_path, 'r', encoding='latin-1') as f:
                email_text = f.read()
            body_text = extract_email_body(email_text)
            email_strip.update({email_file : body_text})
            email_count += 1
            if email_count >= len(labels): #replace with len(labels) to remove limit
                break
        except Exception as e:
            print(f"Error reading {email_file}: {str(e)}")
    
    #preprocessing part
    filtered_emails = extract_words(email_strip)
    keys = email_strip.keys()
    
    return filtered_emails, labels
    '''for keys in email_strip:
        print(keys + ' (' + labels[keys] + ')\n')
        print(email_strip[keys])
        print('\n')'''
    
def extract_words(email_strip, remove_stopwords=True):
    replace_with_space = ['&', '<', '>', '.', ',', ':', ';', '_', '^', '-', '+', '=', '/', '\\', '*', '!', '\'', '"', '(', ')', '[', ']', '}', '{', '?', '$', '#', '@', '|', '%', '\n', '\t']
    remove_completely = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Common email headers that might slip through
    email_artifacts = {'received', 'content', 'type', 'mime', 'version', 'message', 'id', 
                      'subject', 'date', 'from', 'to', 'cc', 'bcc', 'return', 'path'}
    
    stop_words = set()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        stop_words.update(email_artifacts)  # Add email-specific words to remove
    
    for index, text in email_strip.items():
        for char in replace_with_space:
            text = text.replace(char, ' ')
        
        for char in remove_completely:
            text = text.replace(char, '')
        
        words = [
            word 
            for word in text.lower().split() 
            if (word not in stop_words) and (len(word) > 2)
        ]
        email_strip[index] = words
    
    return email_strip


if __name__ == "__main__":
    # Download NLTK stopwords if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    read_and_print_dataset()