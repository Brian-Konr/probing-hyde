#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:31:39 2024

@author: paveenhuang
"""
import os
import torch
import numpy as np
from pathlib import Path
import logging
from model import SAPLMAClassifier
from utils import init_model, load_config, get_probe_path, load_threshold, adjust_probabilities


def get_embedding_from_generation(message, gen_config, model, tokenizer, device, layer=-4):
    """
    Generate text and extract word embeddings from the hidden states.

    Parameters:
    - message (str): The input message from the user.
    - gen_config (dict): Generation configuration parameters.
    - model: The loaded language model.
    - layer: Specific layer of hidden states. 
    - tokenizer: The tokenizer associated with the model.
    - device: The device to perform computations on.

    Returns:
    - dict: Generated text and word embeddings (excluding the input message).
    """
    try:
        if tokenizer.eos_token_id is None:
            if tokenizer.pad_token_id is not None:
                tokenizer.eos_token_id = tokenizer.pad_token_id
        else:
            tokenizer.eos_token = "<EOS>"
            tokenizer.eos_token_id = 50256
            
        # Tokenize the input message
        inputs = tokenizer(message, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].size(1)
        logging.info(f"Input token length: {input_len}")
        
        # Generate text (includes input and generated tokens)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_config,  # Pass gen_config first
                return_dict_in_generate=True,  # Override with desired value
                output_hidden_states=False, # Ensure output_hidden_states is set appropriately
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated token IDs (includes input)
        generated_ids = outputs.sequences[0]  # Shape: (seq_len,)
        generated_text_full = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logging.info(f"Full generated text: {generated_text_full}")
        
        # Re-run the model to get hidden states for the entire sequence
        with torch.no_grad():
            model_outputs = model(generated_ids.unsqueeze(0), output_hidden_states=True)
        
        # Extract hidden states
        all_layer_hidden_states = model_outputs.hidden_states  # Tuple of (num_layers + 1, batch_size, seq_len, hidden_size)
        desired_layer_hidden_states = all_layer_hidden_states[layer]  # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = desired_layer_hidden_states[0]  # Shape: (seq_len, hidden_size)
        logging.info(f"Hidden states shape after extraction: {hidden_states.shape}")
        
        # Convert generated token IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        seq_len = len(tokens)
        logging.info(f"Total number of tokens: {seq_len}")

        if hidden_states.size(0) != seq_len:
            logging.error(f"Mismatch between hidden_states size {hidden_states.size(0)} and tokens length {seq_len}")
            raise ValueError(f"Mismatch between hidden_states size {hidden_states.size(0)} and tokens length {seq_len}")

        # Exclude input tokens to get only the generated part
        generated_ids_only = generated_ids[input_len:]
        hidden_states_only = hidden_states[input_len:]
        tokens_only = tokens[input_len:]
        logging.info(f"Number of generated tokens: {len(tokens_only)}")
        
        # Decode only the generated tokens
        generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True)
        logging.info(f"Generated text (excluding input): {generated_text}")

        # Determine prefix based on tokenizer type
        if hasattr(tokenizer, "word_tokenizer"):
            prefix = "▁"
        else:
            prefix = "Ġ"
        

        # Traverse generated tokens to aggregate word embeddings
        results = []
        current_word_tokens = []
        current_word_embeddings = []

        for j, token in enumerate(tokens_only):
            clean_token = token.lstrip(prefix)
            if prefix and token.startswith(prefix) and current_word_tokens:
                # Aggregate embeddings for the previous word
                word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
                word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
                results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})
                current_word_tokens = [clean_token]
                current_word_embeddings = [hidden_states_only[j]]
            else:
                # Continue building the current word
                current_word_tokens.append(clean_token)
                current_word_embeddings.append(hidden_states_only[j])

        # Process the last word
        if current_word_tokens:
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})

        return {
            "generated_text": generated_text,
            "word_embeddings": results
        }

    except Exception as e:
        logging.exception(f"Error in get_embedding_from_generation: {str(e)}")
        raise e
        
####################################################################################
# {
#     "generated_text": "Hello, how are you today?",
#     "word_embeddings": [
#         {"word": "Hello", "embedding": [0.123, -0.456, ...]},
#         {"word": ",", "embedding": [0.789, -0.012, ...]},
#         {"word": "how", "embedding": [0.345, -0.678, ...]},
#         {"word": "are", "embedding": [0.901, -0.234, ...]},
#         {"word": "you", "embedding": [0.567, -0.890, ...]},
#         {"word": "today", "embedding": [0.345, -0.678, ...]},
#         {"word": "?", "embedding": [0.123, -0.456, ...]}
#     ]
# }
######################################################################################################

# Global cache for probe models
probe_models_cache = {}

def probe_generation(probe_path, probe_name, model, tokenizer, results):
    """
    Perform probing on the generated embeddings.

    Parameters:
    - probe_path (str): Path to the probe model file.
    - probe_name (str): Name of the probe, used for loading the threshold.
    - model: The loaded language model.
    - tokenizer: The tokenizer associated with the model.
    - results (dict): The output from get_embedding_from_generation, containing 'generated_text' and 'word_embeddings'.

    Returns:
    - dict: Updated results with predictions.
    """
    try:
        # Check if the probe model is already loaded
        if probe_path in probe_models_cache:
            probe_model = probe_models_cache[probe_path]
        else:
            input_dim = len(results["word_embeddings"][0]["embedding"])
            probe_model = SAPLMAClassifier(input_dim=input_dim).to(model.device)
            probe_model.load_state_dict(torch.load(probe_path, map_location=model.device))
            probe_model.eval()
            probe_models_cache[probe_path] = probe_model
        
        threshold_file = "threshold.json"
        probe_name = probe_name[:-3]
        threshold = load_threshold(threshold_file, probe_name)
        
        # Prepare embeddings for batch processing
        embeddings = np.array([word_data["embedding"] for word_data in results["word_embeddings"]])
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(model.device)

        # Make predictions in batch
        with torch.no_grad():
            outputs = probe_model(embeddings_tensor)
            probabilities = outputs.cpu().numpy().ravel()
            predictions = (probabilities >= threshold).astype(int)
        
        # Add predictions to words without embedding
        word_predictions = []
        for i, word_data in enumerate(results["word_embeddings"]):
            word_predictions.append({
                "word": word_data["word"],
                "predicted_probability": float(probabilities[i]),
                "predicted_label": int(predictions[i])
            })
            
        data = {
            "generated_text": results["generated_text"],
            "word_predictions": word_predictions
        }
        
        adjusted_data = adjust_probabilities(data)

        return adjusted_data

    except Exception as e:
        logging.exception(f"Error in probe_generation: {str(e)}")
        raise e

####################################################################################
# {
#     "generated_text": "Hello, how are you today?",
#     "word_embeddings": [
#         {
#             "word": "Hello",
#             "embedding": [0.123, -0.456, ...],
#             "predicted_probability": 0.8,
#             "predicted_label": 1
#         }...
# }
####################################################################################


def get_token_embeddings(statement, model, tokenizer, layer, max_seq_length=None, statement_index=0):
    """Process a single statement and get per-token embeddings."""
    results = []
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states[layer]
    input_ids = inputs.input_ids[0]
    attention_mask = inputs.attention_mask[0]
    seq_len = attention_mask.sum().item()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])
    hidden_state = hidden_states[0, :seq_len, :]

    # Check tokenizer type for handling token prefixes
    if hasattr(tokenizer, "word_tokenizer"):
        # For Llama and similar tokenizers
        prefix = "▁"
    else:
        # For GPT-2 and similar tokenizers
        prefix = "Ġ"

    # Traverse each token, aggregate tokens by word and calculate their embeddings
    current_word_tokens = []
    current_word_embeddings = []

    for j, token in enumerate(tokens):
        if prefix and token.startswith(prefix) and current_word_tokens:
            # When encountering a new word, aggregate the token embeddings of the previous word
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})
            current_word_tokens = [token.lstrip("Ġ")]
            current_word_embeddings = [hidden_state[j]]
        else:
            # Continue to add to the current word
            current_word_tokens.append(token.lstrip(prefix))
            current_word_embeddings.append(hidden_state[j])

    # Process the last word
    if current_word_tokens:
        word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
        word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
        results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})

    return results


def probe(statements, model_name, layer, token=None):
    """
    Main function to extract words and their predictions.
    """
    config_parameters = load_config()
    probes_path = Path(config_parameters["probes_dir"])
    token =config_parameters.get("token") # huggingface token
    language_model, tokenizer = init_model(model_name, token)
    max_seq_length = language_model.config.max_position_embeddings
    
    sanitized_model_name = model_name.replace("/", "_")

    all_words_results = []  # Store lists of words for each statement
    

    for idx, statement in enumerate(statements):
        embeddings_data = get_token_embeddings(
            statement, language_model, tokenizer, layer, max_seq_length=max_seq_length, statement_index=idx
        )

        embeddings = np.array([item["embedding"] for item in embeddings_data])
        words = [item["word"] for item in embeddings_data]

        all_words_results.append(words)  # Append word list for each statement

        # Load the trained model path
        probe_model_path = get_probe_path(probes_path, sanitized_model_name, layer)

        # Initialize the probe model with the correct input_dim
        input_dim = embeddings.shape[1]
        probe_model = SAPLMAClassifier(input_dim=input_dim).to(language_model.device)

        # Load the trained weights
        probe_model.load_state_dict(torch.load(probe_model_path, map_location=language_model.device))
        probe_model.eval()

        # Load the optimal threshold
        threshold_file = "threshold.json"
        probe_name = f"{sanitized_model_name}_{abs(layer)}_combined"
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
                "word": word,
                "predicted_probability": probabilities[i],
                "predicted_label": predictions[i],
            }

    return all_words_results  # Return a list of word lists with predictions


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/data1/cache/d12922004"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="embedding_extraction.log")

    model_name = "facebook/opt-350m"
    layer = -4
    statements = [
        "The capital of Australia is Sydney.",
        "Birds lay eggs.",
    ]

    result = probe(statements, model_name, layer)
    print(result)
