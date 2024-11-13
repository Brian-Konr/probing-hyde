#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:58:07 2024

@author: paveenhuang
"""

import sys
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re
# import nltk
from scipy.special import expit  # Sigmoid
from scipy.ndimage import gaussian_filter1d

# nltk.download('punkt_tab')

def load_config(json_path="config.json"):
    """
    Load the configuration file, initialize the model, and process the dataset.
    """
    try:
        with open(json_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return
    except PermissionError:
        logging.error("Permission denied.")
        return
    except json.JSONDecodeError:
        logging.error("Invalid JSON in config file.")
        return


def load_threshold(threshold_file, probe_name):
    """
    Load the optimal threshold from a JSON file.
    """
    try:
        with open(threshold_file, "r") as f:
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


def get_probe_path(probe_path, model_name, layer):
    """
    Get the path of the trained probe model.
    """
    probe_path = probe_path / f"{model_name}_{abs(layer)}_combined.pt"
    if not probe_path.exists():
        logging.error(f"Trained model not found at {probe_path}")
        raise FileNotFoundError(f"Trained model not found at {probe_path}")
    logging.info(f"Trained model found at {probe_path}")
    return probe_path


def select_device():
    """
    Select the appropriate device: GPU (CUDA), Apple MPS, or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def init_model(model_name: str, token: str):
    """
    Initialize the model and tokenizer with automatic device mapping and optional 8-bit quantization.
    Supports both OPT and Llama models.
    """
    try:
        logging.info(f"Loading model: {model_name}")

        # Determine whether it is a Llama model (assuming the Llama model name contains 'llama')
        is_llama = "llama" in model_name.lower()

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, trust_remote_code=is_llama)
        logging.info("Tokenizer loaded successfully.")

        # Set pad_token to eos_token (if the model requires it)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", use_auth_token=token, trust_remote_code=is_llama  # trust remote for llama
        )
        logging.info("Model loaded successfully.")

        return model, tokenizer
    except Exception as e:
        logging.error(f"Model initialization error: {e}")
        sys.exit(1)


def custom_sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def adjust_probabilities(data, alpha=20, c=0.5, threshold = 0.5):
    sentences = custom_sentence_split(data["generated_text"])
    word_predictions = data["word_predictions"]
    
    sentences_word_preds = []
    current_word_idx = 0
    total_preds = len(word_predictions)
    
    for sentence in sentences:
        words = sentence.split()
        num_words = len(words)
        
        if current_word_idx + num_words > total_preds:
            sentence_preds = word_predictions[current_word_idx:]
            current_word_idx = total_preds  
        else:
            last_word = words[-1]
            last_num = current_word_idx + num_words - 1
            while last_num >= current_word_idx and last_word not in word_predictions[last_num]["word"]:
                last_num -= 1
            
            if last_num < current_word_idx:
                sentence_preds = word_predictions[current_word_idx:current_word_idx + num_words]
                current_word_idx += num_words  
            else:
                sentence_preds = word_predictions[current_word_idx:last_num + 1]
                current_word_idx = last_num + 1 
            
        sentences_word_preds.append(sentence_preds)
    
    for sentence_preds in sentences_word_preds:
        L = len(sentence_preds)
        if L == 1:
            continue  
        
        y = np.array([1 - expit(alpha * (i / (L - 1) - c)) for i in range(L)])
        
        original_probs = np.array([wp["predicted_probability"] for wp in sentence_preds])
        
        adjusted_probs = original_probs + y
        adjusted_probs = np.clip(adjusted_probs, 0, 1)  # 保持在 [0, 1] 范围内
        
        smoothed_probs = gaussian_filter1d(adjusted_probs, sigma=1)
        
        for i, wp in enumerate(sentence_preds):
            wp["predicted_probability"] = float(smoothed_probs[i])
            wp["predicted_label"] = int(wp["predicted_probability"] >= threshold)
    
    updated_word_predictions = []
    for sentence_preds in sentences_word_preds:
        for wp in sentence_preds:
            updated_word_predictions.append(wp)
            
    
    data["word_predictions"] = updated_word_predictions
    
    return data

if __name__ == "__main__":
    data = {"generated_text":"?\nI'm not aware of the QS ranking for Nation Taiwan University in CS in 2025, as the QS rankings are typically released annually, and the most recent rankings I have access to are from 2022. For the most up-to-date and accurate information, I recommend checking the QS World University Rankings website or contacting Nation Taiwan University directly.\nHowever, I can provide some general information about Nation Taiwan University's Computer Science program. Nation Taiwan University (NTU) is a","word_predictions":[{"word":"?\nI'm","predicted_probability":0.7541975975036621,"predicted_label":1},{"word":"not","predicted_probability":0.6772081255912781,"predicted_label":1},{"word":"aware","predicted_probability":0.8104518055915833,"predicted_label":1},{"word":"of","predicted_probability":0.7115554213523865,"predicted_label":1},{"word":"the","predicted_probability":0.8192371129989624,"predicted_label":1},{"word":"QS","predicted_probability":0.9778473377227783,"predicted_label":1},{"word":"ranking","predicted_probability":0.9534897208213806,"predicted_label":1},{"word":"for","predicted_probability":0.8381590843200684,"predicted_label":1},{"word":"Nation","predicted_probability":0.9595881104469299,"predicted_label":1},{"word":"Taiwan","predicted_probability":0.8266626596450806,"predicted_label":1},{"word":"University","predicted_probability":0.3353525996208191,"predicted_label":0},{"word":"in","predicted_probability":0.9325249195098877,"predicted_label":1},{"word":"CS","predicted_probability":0.6573321223258972,"predicted_label":1},{"word":"in","predicted_probability":0.5129726529121399,"predicted_label":1},{"word":"2025,","predicted_probability":0.8991391062736511,"predicted_label":1},{"word":"as","predicted_probability":0.7918300628662109,"predicted_label":1},{"word":"the","predicted_probability":0.7788444757461548,"predicted_label":1},{"word":"QS","predicted_probability":0.9561748504638672,"predicted_label":1},{"word":"rankings","predicted_probability":0.998540997505188,"predicted_label":1},{"word":"are","predicted_probability":0.9981493949890137,"predicted_label":1},{"word":"typically","predicted_probability":0.9976796507835388,"predicted_label":1},{"word":"released","predicted_probability":0.7498317360877991,"predicted_label":1},{"word":"annually,","predicted_probability":0.9926525354385376,"predicted_label":1},{"word":"and","predicted_probability":0.5573698282241821,"predicted_label":1},{"word":"the","predicted_probability":0.523878276348114,"predicted_label":1},{"word":"most","predicted_probability":0.29832032322883606,"predicted_label":0},{"word":"recent","predicted_probability":0.6785547137260437,"predicted_label":1},{"word":"rankings","predicted_probability":0.9698278903961182,"predicted_label":1},{"word":"I","predicted_probability":0.8807327747344971,"predicted_label":1},{"word":"have","predicted_probability":0.6999621391296387,"predicted_label":1},{"word":"access","predicted_probability":0.6381951570510864,"predicted_label":1},{"word":"to","predicted_probability":0.9692416191101074,"predicted_label":1},{"word":"are","predicted_probability":0.9569336771965027,"predicted_label":1},{"word":"from","predicted_probability":0.9819836616516113,"predicted_label":1},{"word":"2022.","predicted_probability":0.37548699975013733,"predicted_label":0},{"word":"For","predicted_probability":0.5033490657806396,"predicted_label":1},{"word":"the","predicted_probability":0.6939426064491272,"predicted_label":1},{"word":"most","predicted_probability":0.22997145354747772,"predicted_label":0},{"word":"up-to-date","predicted_probability":0.8046746253967285,"predicted_label":1},{"word":"and","predicted_probability":0.3659253716468811,"predicted_label":0},{"word":"accurate","predicted_probability":0.8747125267982483,"predicted_label":1},{"word":"information,","predicted_probability":0.9638965725898743,"predicted_label":1},{"word":"I","predicted_probability":0.9870918989181519,"predicted_label":1},{"word":"recommend","predicted_probability":0.9817293286323547,"predicted_label":1},{"word":"checking","predicted_probability":0.9147582054138184,"predicted_label":1},{"word":"the","predicted_probability":0.8763933777809143,"predicted_label":1},{"word":"QS","predicted_probability":0.8675969839096069,"predicted_label":1},{"word":"World","predicted_probability":0.7837334871292114,"predicted_label":1},{"word":"University","predicted_probability":0.9451382160186768,"predicted_label":1},{"word":"Rankings","predicted_probability":0.9015120267868042,"predicted_label":1},{"word":"website","predicted_probability":0.7239673137664795,"predicted_label":1},{"word":"or","predicted_probability":0.20760172605514526,"predicted_label":0},{"word":"contacting","predicted_probability":0.8915579319000244,"predicted_label":1},{"word":"Nation","predicted_probability":0.8800478577613831,"predicted_label":1},{"word":"Taiwan","predicted_probability":0.3871750831604004,"predicted_label":0},{"word":"University","predicted_probability":0.09099909663200378,"predicted_label":0},{"word":"directly.\nHowever,","predicted_probability":0.6978898048400879,"predicted_label":1},{"word":"I","predicted_probability":0.9306685328483582,"predicted_label":1},{"word":"can","predicted_probability":0.982677161693573,"predicted_label":1},{"word":"provide","predicted_probability":0.41340258717536926,"predicted_label":0},{"word":"some","predicted_probability":0.6018362045288086,"predicted_label":1},{"word":"general","predicted_probability":0.8636987805366516,"predicted_label":1},{"word":"information","predicted_probability":0.850207507610321,"predicted_label":1},{"word":"about","predicted_probability":0.9880411624908447,"predicted_label":1},{"word":"Nation","predicted_probability":0.9403245449066162,"predicted_label":1},{"word":"Taiwan","predicted_probability":0.8996593952178955,"predicted_label":1},{"word":"University's","predicted_probability":0.8963985443115234,"predicted_label":1},{"word":"Computer","predicted_probability":0.8885548114776611,"predicted_label":1},{"word":"Science","predicted_probability":0.9942763447761536,"predicted_label":1},{"word":"program.","predicted_probability":0.759580135345459,"predicted_label":1},{"word":"Nation","predicted_probability":0.9433199167251587,"predicted_label":1},{"word":"Taiwan","predicted_probability":0.9832646250724792,"predicted_label":1},{"word":"University","predicted_probability":0.986169695854187,"predicted_label":1},{"word":"(NTU)","predicted_probability":0.9987615346908569,"predicted_label":1},{"word":"is","predicted_probability":0.9998736381530762,"predicted_label":1},{"word":"a","predicted_probability":0.9998542070388794,"predicted_label":1}]}
    data_new = adjust_probabilities(data, alpha=20, c=0.5)
    