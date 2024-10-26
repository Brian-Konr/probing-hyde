#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:54:30 2024

@author: paveenhuang
"""

def load_config(config_file):
    """Load configuration from JSON file."""
    try:
        with open(config_file) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        logging.error(f"Config file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON in {config_file}.")
        raise

def init_model(model_name: str):
    """Initialize the language model and tokenizer."""
    try:
        model_name_full = f"facebook/opt-{model_name}"
        logging.info(f"Loading model: {model_name_full}")
        
        model = OPTForCausalLM.from_pretrained(
            model_name_full,
            device_map="auto",
            torch_dtype=torch.float16,  
            load_in_8bit=False  
        )
        logging.info("Model loaded successfully.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_full)
        logging.info("Tokenizer loaded successfully.")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model initialization error: {e}")
        raise

def get_token_embeddings(statement, model, tokenizer, layer, max_seq_length=None, statement_index=0):
    """Process a single statement and get per-token embeddings."""
    results = []
    inputs = tokenizer(statement, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_seq_length, return_attention_mask=True, add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states[layer]
    input_ids = inputs.input_ids[0]  
    attention_mask = inputs.attention_mask[0]  
    seq_len = attention_mask.sum().item()  

    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])
    hidden_state = hidden_states[0, :seq_len, :]  

    current_word_tokens = []
    current_word_embeddings = []

    for j, token in enumerate(tokens):
        if token.startswith('Ġ') and current_word_tokens:
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({'word': word_text, 'embedding': word_embedding.cpu().numpy()})
            current_word_tokens = [token.lstrip('Ġ')]
            current_word_embeddings = [hidden_state[j]]
        else:
            current_word_tokens.append(token.lstrip('Ġ'))
            current_word_embeddings.append(hidden_state[j])

    if current_word_tokens:
        word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
        word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
        results.append({'word': word_text, 'embedding': word_embedding.cpu().numpy()})
        
    return results

def load_trained_model_path(probes_path, model_name, layer):
    """Get the path of the trained probe model."""
    model_path = probes_path / f"{model_name}_{abs(layer)}_combined.pt"
    if not model_path.exists():
        logging.error(f"Trained model not found at {model_path}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    logging.info(f"Trained model found at {model_path}")
    return model_path

def load_threshold(threshold_file, probe_name):
    """Load the optimal threshold from a JSON file."""
    try:
        with open(threshold_file, 'r') as f:
            thresholds = json.load(f)
        threshold = thresholds.get(probe_name, 0.5)
        logging.info(f"Optimal threshold for {probe_name} loaded: {threshold}")
        return threshold
    except FileNotFoundError:
        logging.error(f"Threshold file {threshold_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON in {threshold_file}.")
        raise

def probe(statements, model_name, layer):
    """Main function to extract words and their predictions."""
    config_parameters = load_config("config.json")
    probes_path = Path(config_parameters["probes_dir"])
    language_model, tokenizer = init_model(model_name)
    max_seq_length = language_model.config.max_position_embeddings

    all_words_results = []  # Store lists of words for each statement

    for idx, statement in enumerate(statements):
        embeddings_data = get_token_embeddings(statement, language_model, tokenizer, layer,
                                                max_seq_length=max_seq_length, statement_index=idx)

        embeddings = np.array([item['embedding'] for item in embeddings_data])
        words = [item['word'] for item in embeddings_data]

        all_words_results.append(words)  # Append word list for each statement

        # Load the trained model path
        model_path = load_trained_model_path(probes_path, model_name, layer)

        # Initialize the probe model with the correct input_dim
        input_dim = embeddings.shape[1]
        probe_model = SAPLMAClassifier(input_dim=input_dim).to(language_model.device)

        # Load the trained weights
        probe_model.load_state_dict(torch.load(model_path, map_location=language_model.device))
        probe_model.eval()

        # Load the optimal threshold
        threshold_file = 'threshold.json'
        probe_name = f"{model_name}_{abs(layer)}_combined"
        threshold = load_threshold(threshold_file, probe_name)

        # Convert embeddings to tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(language_model.device)

        # Make predictions
        with torch.no_grad():
            outputs = probe_model(embeddings_tensor)
            probabilities = outputs.cpu().numpy().ravel()
            predictions = (probabilities >= threshold).astype(int)

        # Add predictions to words
        for i, word in enumerate(words):
            all_words_results[idx][i] = {
                'word': word,
                'predicted_probability': probabilities[i],
                'predicted_label': predictions[i]
            }

    return all_words_results  # Return a list of word lists with predictions

if __name__ == "__main__":
    import os
    import torch
    from transformers import AutoTokenizer, OPTForCausalLM
    import numpy as np
    import json
    from pathlib import Path
    import logging
    from model import SAPLMAClassifier  
    
    os.environ['HF_HOME'] = '/data1/cache/d12922004'
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", 
                        filename="embedding_extraction.log")
    
    model_name = '350m'
    layer = -4
    statements = [
        "The capital of Australia is Sydney.",
        # "The Great Wall of China was built entirely during the Qin Dynasty.",
        # "Albert Einstein invented the light bulb.",
        # "Sharks are mammals.",
        # "Mount Everest is located in the United States.",
        # "The moon is larger than Earth.",
        # "The human body has 200 bones.",
        # "Tomatoes are classified as root vegetables.",
        # "The Amazon River flows through South Africa.",
        # "Bats are completely blind.",
        # "Water freezes at 0°C (32°F) under standard atmospheric pressure.",
        # "The Eiffel Tower is located in Paris, France.",
        # "The chemical symbol for water is H₂O.",
        # "Humans have 206 bones in their bodies.",
        # "Venus is the second planet from the Sun.",
        # "The Pacific Ocean is the largest ocean on Earth.",
        # "Whales are mammals.",
        # "The Pythagorean theorem states that a² + b² = c² for right-angled triangles.",
        # "The primary colors are red, blue, and yellow.",
        "Birds lay eggs."
    ]

    result = probe(statements, model_name, layer)
    print(result)  # Output will be in the format [[{word, predicted_probability, predicted_label}, ...], ...]