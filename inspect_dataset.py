import os
import re
from config import *
from collections import defaultdict

def read_and_print_dataset():
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
    print("=== Reading Labels ===")
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
    print(f"Found {len(labels)} emails in labels file.")
    
    # Print sample labels
    #print("\n=== Sample Labels ===")
    #for i, (email_file, label) in enumerate(labels.items()):
    #    if i >= len(labels):
    #        break
    #    print(f"{label.upper()}: {email_file}")
    
    # Read and print sample emails
    #print("\n=== Sample Emails ===")
    email_count = 0
    email_strip = {}
    for email_file in labels.keys():
        email_path = os.path.join(email_dir, email_file)
        try:
            with open(email_path, 'r', encoding='latin-1') as f:
                email_text = f.read()
            email_strip.update({email_file : email_text})
            email_count += 1
            if email_count >= len(labels):
                break
        except Exception as e:
            print(f"Error reading {email_file}: {str(e)}")
    email_strip = extract_words(email_strip)
    keys = email_strip.keys()
    print("\n=== Processed Emails ===\n")
    print(len(keys))
    
def extract_words(email_strip):
    replace_with_space = '&<>.,:;_^-+=/\\*!"()}{?$#@|%\n\t'
    remove_completely = '0123456789'
    
    space_table = str.maketrans(replace_with_space, ' ' * len(replace_with_space))
    remove_table = str.maketrans('', '', remove_completely)
    
    for index, text in email_strip.items():
        text = text.translate(space_table)
        text = text.translate(remove_table)
        text = text.lower().split()
        email_strip[index] = text
    return email_strip

if __name__ == "__main__":
    read_and_print_dataset()