#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:20:52 2024

@author: paveenhuang
"""

import os
import torch
import numpy as np
import gdown
import logging
from model import AttentionMLPSE1DCNN
from utils import load_threshold, adjust_probabilities


def download_model(drive_link, output_path):
    """
    Download the model file from Google Drive.
    Parameters:
    - drive_link (str): drive_link (str): Google Drive shared link.
    - output_path (str): Local save path of the model file.
    """
    try:
        if not os.path.exists(output_path):
            file_id = drive_link.split('/d/')[1].split('/')[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            logging.info(f"Downloading model from {download_url} to {output_path}...")
            gdown.download(download_url, output_path, quiet=False)
            logging.info("Model downloaded successfully.")
        else:
            logging.info(f"Model file already exists at {output_path}. Skipping download.")
    except Exception as e:
        logging.exception(f"Error downloading the model: {str(e)}")
        raise e


def get_embedding_from_generation(message, gen_config, model, tokenizer, device):
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
        
        # Re-run the model to get hidden states for the entire sequence
        with torch.no_grad():
            model_outputs = model(generated_ids.unsqueeze(0), output_hidden_states=True)
        
        all_layer_hidden_states = model_outputs.hidden_states[1:] # skip embedding layer
        num_layers = len(all_layer_hidden_states)
        
        tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        
        # Exclude the input query
        generated_ids_only = generated_ids[input_len:]
        tokens_only = tokens[input_len:]
        generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True)

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
            
            token_all_layers = []
            for layer_idx in range(num_layers):
                token_layer_hs = all_layer_hidden_states[layer_idx][0, j+input_len, :] 
                token_all_layers.append(token_layer_hs)
            token_all_layers = torch.stack(token_all_layers, dim=0)
            
            if prefix and token.startswith(prefix) and current_word_tokens:
                # Aggregate embeddings for the previous word
                word_subtokens = torch.stack(current_word_embeddings, dim=0)
                word_embedding = word_subtokens.mean(dim=0)
                word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
                results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})
                
                # New word
                current_word_tokens = [clean_token]
                current_word_embeddings = [token_all_layers]
            else:
                # Continue building the current word
                current_word_tokens.append(clean_token)
                current_word_embeddings.append(token_all_layers)

        # Process the last word
        if current_word_tokens:
            word_subtokens = torch.stack(current_word_embeddings, dim=0)
            word_embedding = word_subtokens.mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})

        return {
            "generated_text": generated_text,
            "word_embeddings": results
        }

    except Exception as e:
        logging.exception(f"Error in get_embedding_from_generation: {str(e)}")
        raise e


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
            sample_embedding = results["word_embeddings"][0]["embedding"]
            num_layers, hidden_dim = sample_embedding.shape
            
            probe_model = AttentionMLPSE1DCNN(
                hidden_size=hidden_dim,
                num_layers=num_layers,
                num_heads=8,          
                dropout=0.1,          
                reduction=16          
            ).to(model.device)
            
            # load trained weight
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
            outputs = probe_model(embeddings_tensor)  # Shape: (num_words, 1)
            probabilities = outputs.cpu().numpy().ravel() # Shape: (num_words,)
            predictions = (probabilities >= threshold).astype(int) # Shape: (num_words,)
        
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
        
        adjusted_data = data
        # adjusted_data = adjust_probabilities(data)

        return adjusted_data

    except Exception as e:
        logging.exception(f"Error in probe_generation: {str(e)}")
        raise e

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set the model download link and local path
    drive_link = "https://drive.google.com/file/d/15CypPj205eGnBDq7QfiDVXYOArgTQA9Z/view?usp=drive_link"
    model_local_path = "/data2/paveen/probing-hyde/hyde-base/meta-llama_Llama-3.1-8B-Instruct_all-conj_ATTSE1DCNN_head8_dropout0.1.pt"

    download_model(drive_link, model_local_path)

    # model and tokenizer
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # 根据实际情况修改
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()

    # input sample
    message = "What is the capital of France?"
    gen_config = {
        "max_length": 50,
        "num_return_sequences": 1,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95
    }

    # get embedding
    embeddings_results = get_embedding_from_generation(message, gen_config, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")

    # set probe
    probe_path = "/data2/paveen/probing-hyde/hyde-base/meta-llama_Llama-3.1-8B-Instruct_all-conj_ATTSE1DCNN_head8_dropout0.1.pt"
    probe_name = "meta-llama_Llama-3.1-8B-Instruct_all-conj_ATTSE1DCNN_head8_dropout0.1.pt"

    probe_results = probe_generation(probe_path, probe_name, model, tokenizer, embeddings_results)

    # output
    print(probe_results)