import os
import pandas as pd
import re
from transformers import pipeline

# ===================================================================
#  Part 1: Rule-Based Logic (Unaltered)
# ===================================================================
# --- Definitive Lenovo Sensical QA Guidelines ---
INFORMATIONAL_WORDS = ['specifications', 'features', 'availability', 'quality', 'vs', 'performance', 'requirements', 'options', 'solutions', 'configurations', 'prices', 'pricing', 'comparison', 'upgrades', 'compatibility', 'support']
TRANSACTIONAL_WORDS = ['purchases', 'buy', 'sale', 'deals']
SUBJECTIVE_FLUFF = ['stunning', 'dazzling', 'immersive', 'vivid', 'excellent', 'great']
VAGUE_USE_CASES = ['for remote work', 'for photos and movies', 'for home learning', 'for on-the-go', 'for work and play']
UNDESIRABLE_FEATURES = ['glare screen']
BLACKLISTED_WORDS = ['design', 'collaboration', 'durable', 'traditional']
NON_ENGLISH_WORDS = ['para', 'de', 'hombre', 'mujer', 'auriculares', 'funda', 'kabels', 'claviers', 'musmattor', 'ratones', 'toners', 'zoll', 'kaufen', 'förstärkar', 'ordinateur', 'etc.']

APPLE_ECOSYSTEM = ['mac', 'macbook', 'ipad', 'iphone']
VERY_SUBJECTIVE = ['reliable', 'stylish', 'premium', 'chic', 'cool', 'elegant']
NICHE_PROFESSIONAL = ['for cad', 'for avid', 'for engineers', 'for designers', 'for architects', 'for scientists']

INCOMPLETE_PHRASES = ['p series p16s', 'home series tiny', 't series t14s', 'l-series ips']

IRRELEVANT_ITEMS = ['dvd burner', 'thermal printer', 'zagg mophie', 'philips hue', 'stroller', 'car seat']
COMPETITOR_HARDWARE = ['dell xps', 'hp spectre']


def score_keyword_rules(keyword):
    """Applies the definitive, strict rule-based scoring logic."""
    kw_lower = str(keyword).lower()
    if any(word in kw_lower for word in INFORMATIONAL_WORDS): return 0, 'Rule: Informational'
    if any(word in kw_lower for word in TRANSACTIONAL_WORDS): return 0, 'Rule: Transactional'
    if any(word in kw_lower for word in SUBJECTIVE_FLUFF): return 0, 'Rule: Subjective Fluff'
    if any(phrase in kw_lower for phrase in VAGUE_USE_CASES): return 0, 'Rule: Vague Use-Case'
    if any(phrase in kw_lower for phrase in UNDESIRABLE_FEATURES): return 0, 'Rule: Undesirable Feature'
    if any(word in kw_lower for word in BLACKLISTED_WORDS): return 0, 'Rule: Blacklisted Term'
    if any(word in kw_lower.split() for word in NON_ENGLISH_WORDS): return 0, 'Rule: Non-English'
    if kw_lower.startswith('best ') or kw_lower.startswith('top '): return 0, 'Rule: Starts with "best" or "top"'
    if "no operating system" in kw_lower: return 0, 'Rule: Specifies No OS'
    if any(item in kw_lower for item in IRRELEVANT_ITEMS): return -1, 'Rule: Irrelevant Category'
    if any(item in kw_lower for item in COMPETITOR_HARDWARE): return -1, 'Rule: Competitor Hardware'
    if any(word in kw_lower for word in APPLE_ECOSYSTEM): return 0.5, 'Rule: Apple-related Keyword'
    if any(word in kw_lower for word in VERY_SUBJECTIVE): return 0.5, 'Rule: Very Subjective'
    if any(phrase in kw_lower for phrase in NICHE_PROFESSIONAL): return 0.5, 'Rule: Niche Professional Use-Case'
    if re.search(r'(laptops|notebooks|desktops|pcs|computers|workstations|monitors|headsets|keyboards|mice)\s(\d+gb|\d+tb|\d+wh|\d+nits)', kw_lower): return 2, 'Rule: Missing Conjunction'
    if re.search(r'thinkpad\s\w+\s\d+gb', kw_lower): return 2, 'Rule: Incomplete Phrase'
    if kw_lower in INCOMPLETE_PHRASES: return 2, 'Rule: Incomplete Phrase'
    return 1, 'Passed Rules'

# ===================================================================
#  Part 2: Model Loading and Hybrid Classification (Unaltered)
# ===================================================================

def load_model(model_path):
    """Loads the Hugging Face text-classification pipeline."""
    print("Loading the keyword classification model...")
    try:
        classifier = pipeline("text-classification", model=model_path)
        print("Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"Fatal Error loading model: {e}")
        exit()

def hybrid_classify_keyword(keyword, classifier):
    """Applies the hybrid waterfall logic."""
    rule_score, rule_reason = score_keyword_rules(keyword)
    if rule_score != 1:
        return rule_score, rule_reason
    try:
        result = classifier(keyword)[0]
        model_prediction = int(result['label'].split('_')[1])
        model_confidence = result['score']
        model_verdict = "Non-Sensical" if model_prediction == 0 else "Sensical"
        final_reason = f"Model: {model_verdict} (Confidence: {model_confidence:.2%})"
        return model_prediction, final_reason
    except Exception as e:
        return -99, f"Model Error: {e}"

# ===================================================================
#  Part 3: Evaluation and Metrics Reporting
# ===================================================================

def main():
    """
    Main function to run the evaluation of the Hybrid Sensical QA tool.
    """
    # --- Configuration ---
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "sensical_keyword_classifier")
    SENSICAL_FILE = 'sensical.csv'
    NONSENSICAL_FILE = 'non_sensical.csv'
    OUTPUT_FILE = 'hybrid_evaluation_results.csv'
    KEYWORD_COLUMN = 'keyword' # Column name in your test files

    # --- Load Model ---
    classifier = load_model(MODEL_PATH)

    # --- Load and Prepare Test Data ---
    print(f"\nLoading keywords from '{SENSICAL_FILE}' and '{NONSENSICAL_FILE}'...")
    try:
        sensical_df = pd.read_csv(SENSICAL_FILE)
        sensical_df['true_label'] = 1
        nonsensical_df = pd.read_csv(NONSENSICAL_FILE)
        nonsensical_df['true_label'] = 0
        all_keywords_df = pd.concat([sensical_df, nonsensical_df], ignore_index=True)
        keywords_to_test = all_keywords_df[KEYWORD_COLUMN].tolist()
        print(f"Loaded {len(keywords_to_test)} total keywords for evaluation.")
    except FileNotFoundError as e:
        print(f"\nError: Could not find file -> {e.filename}")
        print("Please make sure 'sensical.csv' and 'non_sensical.csv' are in the folder.")
        exit()

    # --- Run Classification and Collect Results ---
    print("\nClassifying keywords with the hybrid system...")
    all_results = []
    for index, row in all_keywords_df.iterrows():
        keyword = row[KEYWORD_COLUMN]
        true_label = row['true_label']
        
        predicted_score, reason = hybrid_classify_keyword(keyword, classifier)
        
        # Convert predicted score to a binary label for accuracy check
        # 1 is sensical, anything else is non-sensical.
        binary_prediction = 1 if predicted_score == 1 else 0
        
        is_correct = (binary_prediction == true_label)
        status = "✅ Correct" if is_correct else "❌ INCORRECT"
        
        all_results.append([keyword, true_label, predicted_score, reason, status])

    # --- Create and Save Detailed Results DataFrame ---
    results_df = pd.DataFrame(
        all_results,
        columns=['Keyword', 'True_Label', 'Predicted_Score', 'Reason', 'Status']
    )
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Detailed analysis complete. Results saved to '{OUTPUT_FILE}'.")

    # --- Calculate and Display Final Accuracy Summary ---
    num_sensical = len(sensical_df)
    num_nonsensical = len(nonsensical_df)
    total_sensical_correct = len(results_df[(results_df['True_Label'] == 1) & (results_df['Status'] == '✅ Correct')])
    total_nonsensical_correct = len(results_df[(results_df['True_Label'] == 0) & (results_df['Status'] == '✅ Correct')])
    total_keywords = len(all_keywords_df)
    total_correct = total_sensical_correct + total_nonsensical_correct

    sensical_accuracy = (total_sensical_correct / num_sensical * 100) if num_sensical > 0 else 0
    nonsensical_accuracy = (total_nonsensical_correct / num_nonsensical * 100) if num_nonsensical > 0 else 0
    overall_accuracy = (total_correct / total_keywords * 100) if total_keywords > 0 else 0

    print("\n==============================================")
    print("      HYBRID CLASSIFIER ACCURACY SUMMARY")
    print("==============================================")
    print(f"Sensical Keywords:     {total_sensical_correct}/{num_sensical} correct ({sensical_accuracy:.1f}%)")
    print(f"Non-sensical Keywords: {total_nonsensical_correct}/{num_nonsensical} correct ({nonsensical_accuracy:.1f}%)")
    print("----------------------------------------------")
    print(f"Overall Accuracy:      {total_correct}/{total_keywords} correct ({overall_accuracy:.1f}%)")
    print("==============================================\n")

if __name__ == '__main__':
    main()
