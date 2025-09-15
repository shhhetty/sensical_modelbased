# ===================================================================
#  Keyword Classification Demo with Accuracy Calculation
# ===================================================================

import os
import pandas as pd
from transformers import pipeline

# --- 1. DEFINE FILE PATHS ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sensical_keyword_classifier")
SENSICAL_FILE = 'sensical.csv'
NONSENSICAL_FILE = 'non_sensical.csv'

# --- 2. LOAD THE MODEL ---
print("Loading the keyword classification model...")
try:
    classifier = pipeline("text-classification", model=MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Provide more specific troubleshooting for this script
    print("\nTroubleshooting:")
    print("1. Make sure the 'sensical_keyword_classifier' folder is in the same directory as this script.")
    print("2. Make sure you have run 'pip install transformers torch pandas'.")
    exit()

# --- 3. LOAD AND PREPARE TEST DATA FROM CSV FILES ---
print(f"\nLoading keywords from '{SENSICAL_FILE}' and '{NONSENSICAL_FILE}'...")
try:
    # Load sensical keywords and assign their true label as 1
    sensical_df = pd.read_csv(SENSICAL_FILE)
    sensical_df['true_label'] = 1

    # Load non-sensical keywords and assign their true label as 0
    nonsensical_df = pd.read_csv(NONSENSICAL_FILE)
    nonsensical_df['true_label'] = 0

    # Combine them into a single DataFrame for processing
    all_keywords_df = pd.concat([sensical_df, nonsensical_df], ignore_index=True)
    
    # Get a list of the keywords to feed into the classifier
    keywords_to_test = all_keywords_df['keyword'].tolist()
    
    print(f"Loaded {len(keywords_to_test)} total keywords for evaluation.")

except FileNotFoundError as e:
    print(f"\nError: Could not find file -> {e.filename}")
    print("Please make sure 'sensical.csv' and 'non_sensical.csv' are in the same folder as this script.")
    exit()

# --- 4. RUN THE CLASSIFICATION ---
print("\nClassifying keywords...")
results = classifier(keywords_to_test)
print("\n--- INDIVIDUAL RESULTS ---")

# --- 5. PROCESS RESULTS AND CALCULATE SCORES ---
total_sensical_correct = 0
total_nonsensical_correct = 0

# Loop through the original dataframe rows and the model results at the same time
for (index, row), result in zip(all_keywords_df.iterrows(), results):
    keyword = row['keyword']
    true_label = row['true_label']
    
    predicted_class = int(result['label'].split('_')[1])
    confidence = result['score']
    
    is_correct = (predicted_class == true_label)
    
    # Increment the correct counters
    if is_correct:
        if true_label == 1:
            total_sensical_correct += 1
        else:
            total_nonsensical_correct += 1
            
    # Display the result for each keyword with a clear status marker
    status = "✅ Correct" if is_correct else "❌ INCORRECT"
    print(f"Keyword:    '{keyword}'")
    print(f"Prediction: {predicted_class} (True Label: {true_label})")
    print(f"Status:     {status}")
    print(f"Confidence: {confidence:.2%}\n")


# --- 6. DISPLAY THE FINAL ACCURACY SUMMARY ---
num_sensical = len(sensical_df)
num_nonsensical = len(nonsensical_df)
total_keywords = len(all_keywords_df)
total_correct = total_sensical_correct + total_nonsensical_correct

# Calculate percentages, avoiding division by zero if a file is empty
sensical_accuracy = (total_sensical_correct / num_sensical * 100) if num_sensical > 0 else 0
nonsensical_accuracy = (total_nonsensical_correct / num_nonsensical * 100) if num_nonsensical > 0 else 0
overall_accuracy = (total_correct / total_keywords * 100) if total_keywords > 0 else 0

print("\n=======================================")
print("        DEMO ACCURACY SUMMARY")
print("=======================================")
print(f"Sensical Keywords:     {total_sensical_correct}/{num_sensical} correct ({sensical_accuracy:.1f}%)")
print(f"Non-sensical Keywords: {total_nonsensical_correct}/{num_nonsensical} correct ({nonsensical_accuracy:.1f}%)")
print("---------------------------------------")
print(f"Overall Accuracy:      {total_correct}/{total_keywords} correct ({overall_accuracy:.1f}%)")
print("=======================================\n")
