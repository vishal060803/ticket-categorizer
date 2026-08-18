# Auto Email/Ticket Categorizer — Project Plan

## 1) Project Goal
Build a lightweight internal triage tool that reads incoming support ticket text (subject + body) and automatically assigns one category:
- Billing
- Technical
- HR
- General

The system should be fast, explainable, and safe for real queue usage, with confidence-aware routing to human review when uncertain.

## 2) Scope (This Build)
### Core Scope
1. Load a labeled dataset of tickets/emails.
2. Clean and preprocess text.
3. Convert text to numerical features (TF-IDF or Bag-of-Words).
4. Train a text classifier (Naive Bayes or Logistic Regression).
5. Evaluate performance using multiple metrics.
6. Run predictions on at least 5 new unseen sample tickets.

### Bonus Scope
1. Output confidence score (probability).
2. Add low-confidence fallback: Needs human review under threshold (~60%).
3. Add urgency tag (urgent/normal) with keyword rules.
4. Provide mini live demo interface (CLI or Streamlit).
5. Add short reflection note on future improvements.

## 3) Data Inputs and Assumptions
- Input records contain at least:
  - `subject`
  - `body`
  - `category` (label)
- Labels are restricted to: Billing, Technical, HR, General.
- `subject` and `body` will be concatenated into a single model text field for training and inference.
- Dataset is expected to be small-to-medium; approach prioritizes speed and reliability over deep learning complexity.

## 4) Preprocessing Plan
1. Combine `subject + body`.
2. Normalize text:
   - lowercase
   - trim extra spaces
   - remove punctuation/noise tokens as needed
3. Remove stopwords (standard English list, with optional customization).
4. Optional lightweight normalization:
   - stemming or lemmatization (only if it improves validation performance)
5. Preserve high-signal tokens (e.g., invoice, refund, password, payroll, interview).

Why this matters:
- Reduces noise.
- Improves vocabulary quality.
- Helps model generalize on short, messy tickets.

## 5) Feature Engineering Plan
### Primary Representation
- TF-IDF vectors over unigrams (and optionally bigrams).

### Why TF-IDF
- Captures term importance by down-weighting very common words.
- Works very well on sparse text classification tasks.
- Fast for training and inference in production-like queues.

### Alternative/Fallback
- Simple Bag-of-Words / keyword count vectors for baseline comparison.

## 6) Modeling Plan
### Candidate Models
1. Multinomial Naive Bayes
2. Logistic Regression

### Selection Logic
- Start with Naive Bayes as a strong and very fast text baseline.
- Train Logistic Regression for potentially better decision boundaries and calibrated class separation.
- Compare using validation metrics and choose the most reliable model.

### Decision Criteria
- Accuracy is good but not sufficient alone.
- Prefer model with better per-class precision/recall (especially for Technical and Billing misroutes).
- Review confusion matrix for systematic confusion patterns.

## 7) Evaluation Plan
### Metrics
1. Accuracy
2. Precision (per class)
3. Recall (per class)
4. F1-score
5. Confusion matrix

### Interpretation Goals
- Detect if one class dominates predictions.
- Identify costly errors (e.g., Technical issues incorrectly sent to General).
- Confirm balanced behavior across all four labels.

### Validation Method
- Train/test split (stratified) for initial build.
- Optional k-fold cross-validation if dataset size permits.

## 8) Inference & Real-Time Triage Behavior
For each new ticket:
1. Apply same preprocessing pipeline.
2. Vectorize with fitted feature transformer.
3. Predict category and class probabilities.
4. Return:
   - predicted category
   - confidence score (%)
   - routing decision

### Human Review Rule (Bonus)
- If top confidence < 60%:
  - Route to `Needs Human Review`
  - Do not auto-assign final category in queue

This creates a defensible safety layer for ambiguous tickets.

## 9) Priority Tagging Layer (Bonus)
Add a parallel rule-based urgency tag:
- `urgent` if text includes terms like: down, urgent, asap, outage, cannot access, not working
- otherwise `normal`

Output should include both:
- `category` (classification)
- `priority_tag` (rule-based)

This enables operations teams to sort by urgency independent of category.

## 10) Demo Plan (Bonus)
### Option A: CLI Demo
- Prompt for subject/body text.
- Print category, confidence, human-review flag, and priority tag.

### Option B: Streamlit Demo
- Input box + button.
- Show prediction card with confidence and routing decision.
- Highlight low-confidence results in warning style.

## 11) Deliverables
1. Reproducible training script/notebook.
2. Saved preprocessing + model pipeline artifact.
3. Evaluation report (metrics + confusion matrix).
4. Predictions for 5+ new manually written sample tickets.
5. Optional CLI/Streamlit demo.
6. Short reflection note (3–5 lines): what to improve with more data/time.

## 12) Risks & Mitigations
### Risk: Small dataset / class imbalance
- Mitigation: stratified split, class-aware metric review, optional class weighting.

### Risk: Ambiguous ticket language
- Mitigation: confidence threshold + human review queue.

### Risk: Vocabulary drift over time
- Mitigation: periodic retraining with newly labeled tickets.

### Risk: Overfitting to keywords
- Mitigation: validate on unseen examples and monitor per-class recall.

## 13) Success Criteria
Project is successful when:
1. Model predicts all four categories with reliable performance.
2. Evaluation includes accuracy + precision/recall/F1 + confusion matrix.
3. New unseen tickets can be classified instantly.
4. Low-confidence tickets are safely routed to manual review.
5. Tool outputs are clear enough for real operational use.

## 14) Submission Format Note
Submit the final project using this exact structure:

### GitHub / Colab / file link
Paste a link to your notebook, repo, or shared code file.
Example: `https://github.com/your-username/ticket-categorizer`

### Approach summary (2–4 lines)
Briefly describe model choice and how edge cases were handled.
Example: Used TF-IDF + Multinomial Naive Bayes, and applied a confidence threshold so low-certainty tickets route to human review.
