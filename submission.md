# Submission Package

## GitHub / Colab / file link
Paste your final shared link here:
`https://github.com/your-username/ticket-categorizer`

(If submitting from local files, provide your zipped project or shared drive link instead.)

## Approach summary (2–4 lines)
Built a ticket categorization pipeline using TF-IDF text features with two baseline classifiers: Multinomial Naive Bayes and Logistic Regression. Text is preprocessed by merging subject+body, lowercasing, removing noise/punctuation, normalizing spaces, and removing stopwords. Model selection is based on weighted F1, macro recall, and accuracy (not accuracy alone). For production safety, inference returns confidence %, adds urgent/normal priority tagging, and routes low-confidence tickets (<60%) to human review.

## Reflection note (3–5 lines)
With more data, I would improve class balance and add many more real-world phrasing variations per category to reduce ambiguity errors. I would also tune TF-IDF and model hyperparameters using cross-validation, then calibrate probabilities for more reliable confidence thresholds. Next, I would add active-learning feedback from human-reviewed tickets to continuously retrain the model. Finally, I would introduce monitoring for drift and per-class recall so routing quality stays stable over time.
