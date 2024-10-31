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
