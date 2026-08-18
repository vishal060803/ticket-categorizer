0–3 min: Setup and objective lock
- Confirm dataset columns: subject, body, category
- Confirm 4 labels: Billing, Technical, HR, General
- Define final outputs: metrics, 5 sample predictions, confidence + review flag + priority tag

3–8 min: Data preparation plan
- Merge subject + body into one text field
- Apply cleaning rules: lowercase, remove noise/punctuation, normalize spaces, remove stopwords
- Quick sanity check for null/empty tickets and label consistency

8–14 min: Feature + model build plan
- Vectorization: TF-IDF (unigrams, optional bigrams if fast)
- Train baseline model: Multinomial Naive Bayes
- Train comparison model: Logistic Regression
- Select best model based on validation metrics (not only accuracy)

14–20 min: Evaluation plan
- Use stratified train/test split
- Capture: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Note common misclassification patterns and risky confusion pairs

20–24 min: Real-time inference logic plan
- Define prediction output schema:
  - predicted category
  - confidence %
  - Needs Human Review if confidence < 60%
  - priority tag (urgent/normal) via keyword rules

24–27 min: New sample tickets plan
- Write at least 5 unseen sample tickets (mixed categories + 1 ambiguous case)
- Run predicted label + confidence + review decision + priority tag for each

27–30 min: Submission packaging plan
- Prepare project link section (GitHub/Colab/file)
- Write 2–4 line approach summary
- Add 3–5 line reflection note (“what to improve with more data/time”)