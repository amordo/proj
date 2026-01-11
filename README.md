# 🇸🇪 Swedish Grammar Learning App

An interactive web application for learning Swedish grammar using **in-context learning** with transformer models. Provide example corrections, and the AI learns the pattern to correct new sentences—no fine-tuning required!

Try it https://huggingface.co/spaces/salemmmm/proj

## 🎯 What is In-Context Learning?

In-context learning allows language models to learn from examples provided directly in the prompt, without modifying the model's weights. Here's how it works:

1. **Provide Examples**: You give 2-3 pairs of incorrect→correct Swedish sentences
2. **Pattern Recognition**: The model identifies the grammar pattern across your examples
3. **Apply Pattern**: The model applies the same correction logic to your new sentence
4. **Instant Feedback**: Get corrections with explanations based on the pattern

### Example Flow:

```
Example 1:
Incorrect: Jag alltid äter frukost.
Correct: Jag äter alltid frukost.
Rule: Verb must be in second position.

Example 2:
Incorrect: På morgonen jag dricker kaffe.
Correct: På morgonen dricker jag kaffe.
Rule: After time expressions, verb comes before subject.

Your Query:
Incorrect: Idag jag ska gå hem.
→ Model learns the V2 pattern and corrects: "Idag ska jag gå hem."
```

## ⚙️ Configuration

You can modify these settings in `app.py`:

```python
# Change model
MODEL_NAME = "google/flan-t5-base"  # Try: flan-t5-large for better quality

# Adjust generation parameters
max_length=128,      # Maximum output length
num_beams=5,         # Beam search width (higher = better but slower)
```
