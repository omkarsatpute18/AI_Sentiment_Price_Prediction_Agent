
from transformers import pipeline

MODEL_NAME = "Horizon-Labs/multilingual-sentiment-small"

print("Loading multilingual sentiment model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1
)

test_sentences = [
    "I am very happy with this company's performance.",
    "The stock price is falling badly.",
    "The company announced a new AI strategy."
]

print("\nTesting sentiment predictions...\n")

predictions = classifier(
    test_sentences,
    truncation=True,
    max_length=512
)

for sentence, prediction in zip(test_sentences, predictions):
    print("Text:", sentence)
    print("Sentiment:", prediction["label"])
    print("Confidence:", round(prediction["score"], 4))
    print("-" * 50)
