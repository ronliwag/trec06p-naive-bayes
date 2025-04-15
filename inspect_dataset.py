import os
from config import *
from collections import defaultdict

def read_and_print_dataset(max_emails_to_print=6):
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
    print("\n=== Sample Labels ===")
    for i, (email_file, label) in enumerate(labels.items()):
        if i >= max_emails_to_print:
            break
        print(f"{label.upper()}: {email_file}")
    
    # Read and print sample emails
    print("\n=== Sample Emails ===")
    email_count = 0
    for email_file in labels.keys():
        email_path = os.path.join(email_dir, email_file)
        try:
            with open(email_path, 'r', encoding='latin-1') as f:
                email_text = f.read()
            print(f"\n--- {email_file} ({labels[email_file].upper()}) ---")
            print(email_text[:1000] + "...")  # Print first 1000 chars to avoid clutter
            email_count += 1
            if email_count >= max_emails_to_print:
                break
        except Exception as e:
            print(f"Error reading {email_file}: {str(e)}")

if __name__ == "__main__":
    read_and_print_dataset()