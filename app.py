import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load mT5-base model for Swedish grammar correction
MODEL_NAME = "google/mt5-base"
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# Pre-populated grammar pattern templates
GRAMMAR_TEMPLATES = {
    "V2 Word Order (Verb-Second Rule)": {
        "ex1_incorrect": "Jag alltid äter frukost klockan sju.",
        "ex1_correct": "Jag äter alltid frukost klockan sju.",
        "ex1_rule": "In main clauses, the verb must be in second position. The adverb 'alltid' comes after the verb.",
        "ex2_incorrect": "På morgonen jag dricker kaffe.",
        "ex2_correct": "På morgonen dricker jag kaffe.",
        "ex2_rule": "When a time expression starts the sentence, the verb must be second, before the subject.",
        "ex3_incorrect": "Idag vi åker till Stockholm.",
        "ex3_correct": "Idag åker vi till Stockholm.",
        "ex3_rule": "With time adverbs in first position, the verb must be second, followed by the subject.",
        "query": "I helgen jag ska besöka mina föräldrar."
    },
    "Definite Forms (en-words)": {
        "ex1_incorrect": "Kan du ge mig bok?",
        "ex1_correct": "Kan du ge mig boken?",
        "ex1_rule": "En-words need the definite suffix -en when referring to a specific item.",
        "ex2_incorrect": "Jag såg en hund. Hund var svart.",
        "ex2_correct": "Jag såg en hund. Hunden var svart.",
        "ex2_rule": "When referring back to a previously mentioned en-word, add -en suffix for definiteness.",
        "ex3_incorrect": "Flickan läser tidning på tåget.",
        "ex3_correct": "Flickan läser tidningen på tåget.",
        "ex3_rule": "Specific items require the definite form: tidning + -en = tidningen.",
        "query": "Jag gillar bil som står där."
    },
    "Gender Agreement (en/ett + Adjectives)": {
        "ex1_incorrect": "Ett stor hus",
        "ex1_correct": "Ett stort hus",
        "ex1_rule": "Ett-words require adjectives to take the -t ending in indefinite singular form.",
        "ex2_incorrect": "En gammal bord",
        "ex2_correct": "Ett gammalt bord",
        "ex2_rule": "Bord is an ett-word, requiring 'ett' article and adjective ending -t.",
        "ex3_incorrect": "En rött äpple",
        "ex3_correct": "Ett rött äpple",
        "ex3_rule": "Äpple is an ett-word, so use 'ett' and adjective with -t ending.",
        "query": "En vackert land"
    },
    "Verb Conjugation (Strong Verbs Past Tense)": {
        "ex1_incorrect": "Hon drinkade mycket vatten.",
        "ex1_correct": "Hon drack mycket vatten.",
        "ex1_rule": "Dricka (to drink) is a strong verb. Past tense: drack (not drinkade).",
        "ex2_incorrect": "Vi ätade lunch tillsammans.",
        "ex2_correct": "Vi åt lunch tillsammans.",
        "ex2_rule": "Äta (to eat) is a strong verb with vowel change in past tense: åt.",
        "ex3_incorrect": "De skrivade brev till sina föräldrar.",
        "ex3_correct": "De skrev brev till sina föräldrar.",
        "ex3_rule": "Skriva is a strong verb. Past tense: skrev (not skrivade).",
        "query": "Jag gåde till affären igår."
    },
    "Adjective Declension with Definiteness": {
        "ex1_incorrect": "Den stor huset",
        "ex1_correct": "Det stora huset",
        "ex1_rule": "With ett-words in definite form: det + adjective with -a + noun with -et suffix.",
        "ex2_incorrect": "De gammal bilar",
        "ex2_correct": "De gamla bilarna",
        "ex2_rule": "Plural definite: de + adjective with -a + noun with plural definite suffix.",
        "ex3_incorrect": "Den ny boken",
        "ex3_correct": "Den nya boken",
        "ex3_rule": "With definite en-words, the adjective takes -a ending.",
        "query": "Det vacker huset"
    }
}


def format_prompt(ex1_inc, ex1_corr, ex1_rule, ex2_inc, ex2_corr, ex2_rule, 
                  ex3_inc, ex3_corr, ex3_rule, query):
    """Format the in-context learning prompt for grammar correction."""
    
    prompt = "Correct the Swedish grammar errors based on the examples below.\n\n"
    
    # Add examples
    if ex1_inc and ex1_corr:
        prompt += f"Incorrect: {ex1_inc}\n"
        prompt += f"Correct: {ex1_corr}\n"
        if ex1_rule:
            prompt += f"Rule: {ex1_rule}\n"
        prompt += "\n"
    
    if ex2_inc and ex2_corr:
        prompt += f"Incorrect: {ex2_inc}\n"
        prompt += f"Correct: {ex2_corr}\n"
        if ex2_rule:
            prompt += f"Rule: {ex2_rule}\n"
        prompt += "\n"
    
    if ex3_inc and ex3_corr:
        prompt += f"Incorrect: {ex3_inc}\n"
        prompt += f"Correct: {ex3_corr}\n"
        if ex3_rule:
            prompt += f"Rule: {ex3_rule}\n"
        prompt += "\n"
    
    # Add the query
    prompt += f"Now correct this sentence:\n"
    prompt += f"Incorrect: {query}\n"
    prompt += f"Correct:"
    
    return prompt


def generate_correction(ex1_inc, ex1_corr, ex1_rule, ex2_inc, ex2_corr, ex2_rule,
                       ex3_inc, ex3_corr, ex3_rule, query):
    """Generate grammar correction using in-context learning."""
    
    if not query:
        return "Please provide a sentence to correct."
    
    # Format the prompt
    prompt = format_prompt(ex1_inc, ex1_corr, ex1_rule, ex2_inc, ex2_corr, ex2_rule,
                          ex3_inc, ex3_corr, ex3_rule, query)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            early_stopping=True
        )
    
    # Decode the output
    correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Format the response
    response = f"**Corrected Sentence:**\n{correction}\n\n"
    response += f"**Your Input:**\n{query}\n\n"
    response += "**In-Context Learning Prompt Used:**\n```\n{}\n```".format(prompt)
    
    return response


def load_template(template_name):
    """Load a pre-populated grammar template."""
    if template_name in GRAMMAR_TEMPLATES:
        template = GRAMMAR_TEMPLATES[template_name]
        return (
            template["ex1_incorrect"],
            template["ex1_correct"],
            template["ex1_rule"],
            template["ex2_incorrect"],
            template["ex2_correct"],
            template["ex2_rule"],
            template["ex3_incorrect"],
            template["ex3_correct"],
            template["ex3_rule"],
            template["query"]
        )
    return ("", "", "", "", "", "", "", "", "", "")


# Create Gradio interface
with gr.Blocks(title="Swedish Grammar Learning with In-Context Learning") as demo:
    gr.Markdown("""
    # 🇸🇪 Swedish Grammar Learning App
    ## In-Context Learning for Grammar Correction
    
    This app uses **in-context learning** to correct Swedish grammar. Provide 2-3 example corrections 
    demonstrating a specific grammar rule, then submit an incorrect sentence. The AI learns the pattern 
    from your examples and applies it to correct your query—no fine-tuning required!
    
    ### How it works:
    1. Select a grammar template or create your own examples
    2. Provide 2-3 pairs of incorrect→correct sentences with grammar rules
    3. Enter a sentence you want corrected
    4. The model recognizes the pattern and generates the correction
    """)
    
    with gr.Row():
        template_dropdown = gr.Dropdown(
            choices=list(GRAMMAR_TEMPLATES.keys()),
            label="📚 Load Grammar Template (Optional)",
            value=None
        )
        load_btn = gr.Button("Load Template")
    
    gr.Markdown("### Example 1")
    with gr.Row():
        ex1_incorrect = gr.Textbox(label="Incorrect Sentence", placeholder="Jag alltid äter frukost...")
        ex1_correct = gr.Textbox(label="Correct Sentence", placeholder="Jag äter alltid frukost...")
    ex1_rule = gr.Textbox(label="Grammar Rule Explanation", placeholder="The verb must be in second position...")
    
    gr.Markdown("### Example 2")
    with gr.Row():
        ex2_incorrect = gr.Textbox(label="Incorrect Sentence", placeholder="På morgonen jag dricker...")
        ex2_correct = gr.Textbox(label="Correct Sentence", placeholder="På morgonen dricker jag...")
    ex2_rule = gr.Textbox(label="Grammar Rule Explanation", placeholder="When a time expression starts...")
    
    gr.Markdown("### Example 3 (Optional)")
    with gr.Row():
        ex3_incorrect = gr.Textbox(label="Incorrect Sentence", placeholder="Optional third example...")
        ex3_correct = gr.Textbox(label="Correct Sentence", placeholder="Optional...")
    ex3_rule = gr.Textbox(label="Grammar Rule Explanation", placeholder="Optional...")
    
    gr.Markdown("### Your Sentence to Correct")
    query = gr.Textbox(
        label="Incorrect Sentence to Correct",
        placeholder="Enter a Swedish sentence with grammar errors...",
        lines=2
    )
    
    submit_btn = gr.Button("✨ Correct Grammar", variant="primary")
    
    output = gr.Markdown(label="Correction Result")
    
    # Wire up the template loader
    load_btn.click(
        fn=load_template,
        inputs=[template_dropdown],
        outputs=[ex1_incorrect, ex1_correct, ex1_rule, 
                ex2_incorrect, ex2_correct, ex2_rule,
                ex3_incorrect, ex3_correct, ex3_rule, query]
    )
    
    # Wire up the correction generator
    submit_btn.click(
        fn=generate_correction,
        inputs=[ex1_incorrect, ex1_correct, ex1_rule,
               ex2_incorrect, ex2_correct, ex2_rule,
               ex3_incorrect, ex3_correct, ex3_rule, query],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### Grammar Patterns Available:
    - **V2 Word Order**: Verb-second rule in main clauses
    - **Definite Forms**: Suffixed articles (-en, -et, -a)
    - **Gender Agreement**: en/ett words with adjective endings
    - **Verb Conjugation**: Strong vs weak verbs in past tense
    - **Adjective Declension**: Definite vs indefinite forms
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
