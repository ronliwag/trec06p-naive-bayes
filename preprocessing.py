#cleans each email down to the tokenized unique words for each email body stored in an array

import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from config import *
from collections import defaultdict
from email import message_from_string

_HTML_TAG_PATTERN = re.compile(r'<.*?>')

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

def preprocess_dataset(dir_index=DIR_INDEX): 
    # Paths
    email_dir = DIR_DATA
    label_path = dir_index
    
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
                labels[email_file] = 1 if label.lower() == 'spam' else 0  # Convert to 0/1
            except ValueError:
                print(f"Skipping malformed line: {line}")
    
    # Read sample emails
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
    keys = email_strip.keys()
    filtered_emails = {}
    for key in keys:
        filtered_emails.update({key : extract_words(email_strip[key])})
    
    return filtered_emails, labels
    
def extract_words(email_body, remove_stopwords=True):
    # Common email headers that might slip through
    email_artifacts = {'received', 'content', 'type', 'mime', 'version', 'message', 'id', 
                      'subject', 'date', 'from', 'to', 'cc', 'bcc', 'return', 'path'}
    
    stop_words = set()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
    
    email_body = _HTML_TAG_PATTERN.sub('', email_body)

    words = [
        word.lower() 
        for word in email_body.split() 
        if (word.lower() not in stop_words) and (word.isalpha()) and (len(word) > 3)
    ]
    
    return words


if __name__ == "__main__":
    # Download NLTK stopwords if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    preprocess_dataset()