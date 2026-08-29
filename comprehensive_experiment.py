#!/usr/bin/env python3
"""Comprehensive real experiment with diverse data, multiple seeds, and parameter sweeps.

This script includes:
1. 50+ diverse prompts across 10 domains
2. 10+ adversarial/edge case prompts
3. 5 different random seeds
4. Multiple regularization strengths (λ)
5. Multiple mapping strategies
6. 4 model pairs: 4B→1.7B, 4B→0.6B, 8B→1.7B, 8B→0.6B

Usage:
    python comprehensive_experiment.py
"""

import json
import os
import time
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_PATHS = {
    '0.6B': os.path.expanduser('~/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master'),
    '1.7B': os.path.expanduser('~/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master'),
    '4B': os.path.expanduser('~/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots/master'),
    '8B': os.path.expanduser('~/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master'),
}

MODEL_PAIRS = [
    ('4B', '1.7B'),
    ('4B', '0.6B'),
    ('8B', '1.7B'),
    ('8B', '0.6B'),
]

# ============================================================================
# Diverse Prompts (50+ prompts across 10 domains)
# ============================================================================

DIVERSE_PROMPTS = [
    # Domain: Factual (5)
    {"id": "fact_01", "domain": "factual", "question": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Madrid"], "answer": "Paris"},
    {"id": "fact_02", "domain": "factual", "question": "What is the largest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "Pacific"},
    {"id": "fact_03", "domain": "factual", "question": "What is the speed of light?", "choices": ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "answer": "300,000 km/s"},
    {"id": "fact_04", "domain": "factual", "question": "What is the chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": "Au"},
    {"id": "fact_05", "domain": "factual", "question": "What is the boiling point of water?", "choices": ["90°C", "100°C", "110°C", "120°C"], "answer": "100°C"},
    
    # Domain: Mathematical (5)
    {"id": "math_01", "domain": "mathematical", "question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "answer": "4"},
    {"id": "math_02", "domain": "mathematical", "question": "What is 10 - 5?", "choices": ["3", "4", "5", "6"], "answer": "5"},
    {"id": "math_03", "domain": "mathematical", "question": "What is 3 × 4?", "choices": ["7", "10", "12", "14"], "answer": "12"},
    {"id": "math_04", "domain": "mathematical", "question": "What is √16?", "choices": ["2", "3", "4", "5"], "answer": "4"},
    {"id": "math_05", "domain": "mathematical", "question": "What is 15 ÷ 3?", "choices": ["3", "4", "5", "6"], "answer": "5"},
    
    # Domain: Reasoning (5)
    {"id": "reason_01", "domain": "reasoning", "question": "If all dogs are animals, and all animals need food, then:", "choices": ["All dogs need food", "Some dogs need food", "No dogs need food", "Cannot determine"], "answer": "All dogs need food"},
    {"id": "reason_02", "domain": "reasoning", "question": "If A > B and B > C, then:", "choices": ["A > C", "A < C", "A = C", "Cannot determine"], "answer": "A > C"},
    {"id": "reason_03", "domain": "reasoning", "question": "If it rains, the ground gets wet. The ground is wet. Therefore:", "choices": ["It rained", "It might have rained", "The ground is always wet", "None of the above"], "answer": "It might have rained"},
    {"id": "reason_04", "domain": "reasoning", "question": "All birds can fly. A penguin is a bird. Therefore:", "choices": ["Penguins can fly", "Penguins cannot fly", "The premise is wrong", "None of the above"], "answer": "The premise is wrong"},
    {"id": "reason_05", "domain": "reasoning", "question": "If no A are B, and all C are A, then:", "choices": ["All C are B", "No C are B", "Some C are B", "Cannot determine"], "answer": "No C are B"},
    
    # Domain: Science (5)
    {"id": "sci_01", "domain": "science", "question": "What is H2O?", "choices": ["Oxygen", "Water", "Hydrogen", "Carbon"], "answer": "Water"},
    {"id": "sci_02", "domain": "science", "question": "What planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": "Mercury"},
    {"id": "sci_03", "domain": "science", "question": "What gas do plants absorb?", "choices": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"], "answer": "Carbon Dioxide"},
    {"id": "sci_04", "domain": "science", "question": "What is the largest planet?", "choices": ["Saturn", "Jupiter", "Neptune", "Uranus"], "answer": "Jupiter"},
    {"id": "sci_05", "domain": "science", "question": "What is the hardest natural substance?", "choices": ["Gold", "Iron", "Diamond", "Silver"], "answer": "Diamond"},
    
    # Domain: History (5)
    {"id": "hist_01", "domain": "history", "question": "In which year did WWII end?", "choices": ["1943", "1944", "1945", "1946"], "answer": "1945"},
    {"id": "hist_02", "domain": "history", "question": "Who was the first president of the USA?", "choices": ["Lincoln", "Washington", "Jefferson", "Adams"], "answer": "Washington"},
    {"id": "hist_03", "domain": "history", "question": "When did the French Revolution begin?", "choices": ["1776", "1789", "1799", "1804"], "answer": "1789"},
    {"id": "hist_04", "domain": "history", "question": "Who wrote the Declaration of Independence?", "choices": ["Washington", "Jefferson", "Franklin", "Adams"], "answer": "Jefferson"},
    {"id": "hist_05", "domain": "history", "question": "In which century did Columbus discover America?", "choices": ["14th", "15th", "16th", "17th"], "answer": "15th"},
    
    # Domain: Geography (5)
    {"id": "geo_01", "domain": "geography", "question": "What is the largest continent?", "choices": ["Africa", "Asia", "Europe", "North America"], "answer": "Asia"},
    {"id": "geo_02", "domain": "geography", "question": "What is the capital of Japan?", "choices": ["Seoul", "Beijing", "Tokyo", "Bangkok"], "answer": "Tokyo"},
    {"id": "geo_03", "domain": "geography", "question": "What is the longest river?", "choices": ["Amazon", "Nile", "Mississippi", "Yangtze"], "answer": "Nile"},
    {"id": "geo_04", "domain": "geography", "question": "What is the smallest country?", "choices": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"], "answer": "Vatican City"},
    {"id": "geo_05", "domain": "geography", "question": "What is the deepest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "Pacific"},
    
    # Domain: Language (5)
    {"id": "lang_01", "domain": "language", "question": "What is a synonym for 'happy'?", "choices": ["Sad", "Joyful", "Angry", "Tired"], "answer": "Joyful"},
    {"id": "lang_02", "domain": "language", "question": "What is the plural of 'child'?", "choices": ["Childs", "Children", "Childes", "Childies"], "answer": "Children"},
    {"id": "lang_03", "domain": "language", "question": "What is a verb?", "choices": ["Happy", "Run", "Beautiful", "House"], "answer": "Run"},
    {"id": "lang_04", "domain": "language", "question": "What is a noun?", "choices": ["Run", "Happy", "Dog", "Quickly"], "answer": "Dog"},
    {"id": "lang_05", "domain": "language", "question": "What is an antonym for 'big'?", "choices": ["Large", "Huge", "Small", "Tall"], "answer": "Small"},
    
    # Domain: Common Sense (5)
    {"id": "common_01", "domain": "common_sense", "question": "What do you use to write on paper?", "choices": ["Fork", "Pen", "Hammer", "Spoon"], "answer": "Pen"},
    {"id": "common_02", "domain": "common_sense", "question": "What do you use to cut food?", "choices": ["Spoon", "Fork", "Knife", "Plate"], "answer": "Knife"},
    {"id": "common_03", "domain": "common_sense", "question": "What do you use to see?", "choices": ["Ears", "Eyes", "Nose", "Mouth"], "answer": "Eyes"},
    {"id": "common_04", "domain": "common_sense", "question": "What do you use to listen to music?", "choices": ["Eyes", "Ears", "Nose", "Hands"], "answer": "Ears"},
    {"id": "common_05", "domain": "common_sense", "question": "What do you use to open a door?", "choices": ["Key", "Fork", "Pen", "Spoon"], "answer": "Key"},
    
    # Domain: Technical (5)
    {"id": "tech_01", "domain": "technical", "question": "What does CPU stand for?", "choices": ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Computer Processing Unit"], "answer": "Central Processing Unit"},
    {"id": "tech_02", "domain": "technical", "question": "What does RAM stand for?", "choices": ["Random Access Memory", "Read Access Memory", "Random Average Memory", "Read Average Memory"], "answer": "Random Access Memory"},
    {"id": "tech_03", "domain": "technical", "question": "What does GPU stand for?", "choices": ["Graphics Processing Unit", "General Processing Unit", "Graphics Program Utility", "General Purpose Unit"], "answer": "Graphics Processing Unit"},
    {"id": "tech_04", "domain": "technical", "question": "What does USB stand for?", "choices": ["Universal Serial Bus", "United Serial Bus", "Universal System Bus", "United System Bus"], "answer": "Universal Serial Bus"},
    {"id": "tech_05", "domain": "technical", "question": "What does HTML stand for?", "choices": ["Hyper Text Markup Language", "High Tech Modern Language", "Home Tool Markup Language", "Hyperlink Text Mode Language"], "answer": "Hyper Text Markup Language"},
    
    # Domain: Creative (5)
    {"id": "creat_01", "domain": "creative", "question": "What color is the sky on a clear day?", "choices": ["Green", "Blue", "Red", "Yellow"], "answer": "Blue"},
    {"id": "creat_02", "domain": "creative", "question": "What season comes after winter?", "choices": ["Summer", "Fall", "Spring", "Winter"], "answer": "Spring"},
    {"id": "creat_03", "domain": "creative", "question": "What is the opposite of 'hot'?", "choices": ["Warm", "Cold", "Cool", "Mild"], "answer": "Cold"},
    {"id": "creat_04", "domain": "creative", "question": "What is the opposite of 'dark'?", "choices": ["Dim", "Bright", "Shadow", "Night"], "answer": "Bright"},
    {"id": "creat_05", "domain": "creative", "question": "What is the opposite of 'fast'?", "choices": ["Quick", "Rapid", "Slow", "Speedy"], "answer": "Slow"},
]

# ============================================================================
# Adversarial/Edge Case Prompts (10+ prompts)
# ============================================================================

ADVERSARIAL_PROMPTS = [
    # Ambiguous questions
    {"id": "adv_01", "domain": "adversarial", "question": "What is the best color?", "choices": ["Red", "Blue", "Green", "Yellow"], "answer": "Red", "note": "subjective"},
    {"id": "adv_02", "domain": "adversarial", "question": "What is the meaning of life?", "choices": ["42", "Happiness", "Love", "Unknown"], "answer": "42", "note": "philosophical"},
    
    # Tricky wording
    {"id": "adv_03", "domain": "adversarial", "question": "Which is heavier, a pound of feathers or a pound of gold?", "choices": ["Feathers", "Gold", "Same weight", "Cannot determine"], "answer": "Same weight", "note": "trick question"},
    {"id": "adv_04", "domain": "adversarial", "question": "How many months have 28 days?", "choices": ["1", "2", "6", "12"], "answer": "12", "note": "all months"},
    
    # Contradictory premises
    {"id": "adv_05", "domain": "adversarial", "question": "If the earth is flat, what shape is it?", "choices": ["Round", "Flat", "Square", "Triangle"], "answer": "Flat", "note": "false premise"},
    {"id": "adv_06", "domain": "adversarial", "question": "Since the sun is cold, why do we feel warm?", "choices": ["It's not cold", "We don't", "Miracle", "Reflected heat"], "answer": "It's not cold", "note": "false premise"},
    
    # Numerical edge cases
    {"id": "adv_07", "domain": "adversarial", "question": "What is 0 divided by 0?", "choices": ["0", "1", "Undefined", "Infinity"], "answer": "Undefined", "note": "math edge case"},
    {"id": "adv_08", "domain": "adversarial", "question": "What is the smallest positive integer?", "choices": ["0", "1", "2", "None"], "answer": "1", "note": "edge case"},
    
    # Multiple correct answers
    {"id": "adv_09", "domain": "adversarial", "question": "Which is a fruit? (Select all that apply)", "choices": ["Apple", "Carrot", "Banana", "Potato"], "answer": "Apple", "note": "multiple correct"},
    {"id": "adv_10", "domain": "adversarial", "question": "What is 2 + 2 * 2?", "choices": ["4", "6", "8", "12"], "answer": "6", "note": "order of operations"},
    
    # Context-dependent
    {"id": "adv_11", "domain": "adversarial", "question": "Is a tomato a fruit or vegetable?", "choices": ["Fruit", "Vegetable", "Both", "Neither"], "answer": "Fruit", "note": "context dependent"},
    {"id": "adv_12", "domain": "adversarial", "question": "Can you be done doing something you never started?", "choices": ["Yes", "No", "Maybe", "Paradox"], "answer": "Yes", "note": "logical edge case"},
]

# ============================================================================
# Parameter Configurations
# ============================================================================

PARAM_CONFIGS = {
    "seeds": [42, 123, 456, 789, 1024],
    "lambda_values": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "mapping_strategies": ["native_average", "ridge_simple", "weighted_average"],
    "scoring_methods": ["full_text", "letter_only"],
}

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manage model loading and inference."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer."""
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]
        
        path = MODEL_PATHS[model_name]
        logger.info(f"Loading model {model_name} from {path}")
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model, tokenizer
    
    def unload_model(self, model_name: str):
        """Unload model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")

# ============================================================================
# KV Cache Operations with Parameter Settings
# ============================================================================

def native_average_kv(teacher_kv, n_student_layers: int, 
                      student_head_dim: int = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Average teacher KV layers to match student layers."""
    # Convert DynamicCache to list format
    if hasattr(teacher_kv, 'key_cache'):
        # DynamicCache format
        teacher_k = list(teacher_kv.key_cache)
        teacher_v = list(teacher_kv.value_cache)
    else:
        # Tuple format (batch, heads, seq, dim)
        n_teacher_layers = len(teacher_kv) // 2
        teacher_k = [teacher_kv[i*2] for i in range(n_teacher_layers)]
        teacher_v = [teacher_kv[i*2+1] for i in range(n_teacher_layers)]
    
    n_teacher_layers = len(teacher_k)
    
    # Get dimensions
    teacher_head_dim = teacher_k[0].shape[-1]
    if student_head_dim is None:
        student_head_dim = teacher_head_dim
    
    k_per_student = n_teacher_layers / n_student_layers
    student_k = []
    student_v = []
    
    for s in range(n_student_layers):
        start = int(s * k_per_student)
        end = int((s + 1) * k_per_student)
        end = min(end, n_teacher_layers)
        
        avg_k = torch.stack([teacher_k[i] for i in range(start, end)]).mean(dim=0)
        avg_v = torch.stack([teacher_v[i] for i in range(start, end)]).mean(dim=0)
        
        # Handle dimension mismatch
        if teacher_head_dim != student_head_dim:
            # Simple truncation or padding
            if teacher_head_dim > student_head_dim:
                avg_k = avg_k[..., :student_head_dim]
                avg_v = avg_v[..., :student_head_dim]
            else:
                # Pad with zeros
                pad_size = student_head_dim - teacher_head_dim
                avg_k = torch.nn.functional.pad(avg_k, (0, pad_size))
                avg_v = torch.nn.functional.pad(avg_v, (0, pad_size))
        
        student_k.append(avg_k)
        student_v.append(avg_v)
    
    return student_k, student_v


def ridge_simple_kv(teacher_kv, n_student_layers: int,
                   lambda_val: float = 1e-5,
                   student_head_dim: int = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Simple ridge regression mapping."""
    if hasattr(teacher_kv, 'key_cache'):
        teacher_k = list(teacher_kv.key_cache)
        teacher_v = list(teacher_kv.value_cache)
    else:
        n_teacher_layers = len(teacher_kv) // 2
        teacher_k = [teacher_kv[i*2] for i in range(n_teacher_layers)]
        teacher_v = [teacher_kv[i*2+1] for i in range(n_teacher_layers)]
    
    n_teacher_layers = len(teacher_k)
    
    # Get dimensions
    teacher_head_dim = teacher_k[0].shape[-1]
    if student_head_dim is None:
        student_head_dim = teacher_head_dim
    
    student_k = []
    student_v = []
    
    k_per_student = n_teacher_layers / n_student_layers
    
    for s in range(n_student_layers):
        start = int(s * k_per_student)
        end = int((s + 1) * k_per_student)
        end = min(end, n_teacher_layers)
        
        avg_k = torch.stack([teacher_k[i] for i in range(start, end)]).mean(dim=0)
        avg_v = torch.stack([teacher_v[i] for i in range(start, end)]).mean(dim=0)
        
        reg_k = avg_k * (1 - lambda_val)
        reg_v = avg_v * (1 - lambda_val)
        
        # Handle dimension mismatch
        if teacher_head_dim != student_head_dim:
            if teacher_head_dim > student_head_dim:
                reg_k = reg_k[..., :student_head_dim]
                reg_v = reg_v[..., :student_head_dim]
            else:
                pad_size = student_head_dim - teacher_head_dim
                reg_k = torch.nn.functional.pad(reg_k, (0, pad_size))
                reg_v = torch.nn.functional.pad(reg_v, (0, pad_size))
        
        student_k.append(reg_k)
        student_v.append(reg_v)
    
    return student_k, student_v


def weighted_average_kv(teacher_kv, n_student_layers: int,
                       weights: List[float] = None,
                       student_head_dim: int = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Weighted average of teacher layers."""
    if hasattr(teacher_kv, 'key_cache'):
        teacher_k = list(teacher_kv.key_cache)
        teacher_v = list(teacher_kv.value_cache)
    else:
        n_teacher_layers = len(teacher_kv) // 2
        teacher_k = [teacher_kv[i*2] for i in range(n_teacher_layers)]
        teacher_v = [teacher_kv[i*2+1] for i in range(n_teacher_layers)]
    
    n_teacher_layers = len(teacher_k)
    
    # Get dimensions
    teacher_head_dim = teacher_k[0].shape[-1]
    if student_head_dim is None:
        student_head_dim = teacher_head_dim
    
    if weights is None:
        weights = [1.0 / n_teacher_layers] * n_teacher_layers
    
    student_k = []
    student_v = []
    
    k_per_student = n_teacher_layers / n_student_layers
    
    for s in range(n_student_layers):
        start = int(s * k_per_student)
        end = int((s + 1) * k_per_student)
        end = min(end, n_teacher_layers)
        
        total_weight = sum(weights[start:end])
        if total_weight > 0:
            weighted_k = sum(teacher_k[i] * weights[i] for i in range(start, end)) / total_weight
            weighted_v = sum(teacher_v[i] * weights[i] for i in range(start, end)) / total_weight
        else:
            weighted_k = torch.stack([teacher_k[i] for i in range(start, end)]).mean(dim=0)
            weighted_v = torch.stack([teacher_v[i] for i in range(start, end)]).mean(dim=0)
        
        # Handle dimension mismatch
        if teacher_head_dim != student_head_dim:
            if teacher_head_dim > student_head_dim:
                weighted_k = weighted_k[..., :student_head_dim]
                weighted_v = weighted_v[..., :student_head_dim]
            else:
                pad_size = student_head_dim - teacher_head_dim
                weighted_k = torch.nn.functional.pad(weighted_k, (0, pad_size))
                weighted_v = torch.nn.functional.pad(weighted_v, (0, pad_size))
        
        student_k.append(weighted_k)
        student_v.append(weighted_v)
    
    return student_k, student_v


def create_kv_cache(k_list: List[torch.Tensor], v_list: List[torch.Tensor]):
    """Create a DynamicCache from K and V lists."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(zip(k_list, v_list)):
        cache.update(k, v, layer_idx)
    return cache

# ============================================================================
# Scoring Functions
# ============================================================================

def _get_answer_letter(answer: str, choices: List[str]) -> str:
    """Convert answer value to letter label (A/B/C/D)."""
    idx = choices.index(answer)
    return chr(65 + idx)


def _score_logits_by_letter(tokenizer, logits, n_choices: int = 4) -> List[float]:
    """Score logits by letter tokens (A/B/C/D), returning probabilities."""
    probs = torch.softmax(logits, dim=-1)
    letter_scores = []
    for i in range(n_choices):
        letter = chr(65 + i)
        letter_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        # Use the last token (the letter itself, not the space)
        token_id = letter_ids[-1] if letter_ids else tokenizer.encode(letter, add_special_tokens=False)[0]
        letter_scores.append(probs[token_id].item())
    
    total = sum(letter_scores)
    if total > 0:
        letter_scores = [s / total for s in letter_scores]
    
    return letter_scores


def score_full_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                   question: str, choices: List[str], answer: str) -> Dict:
    """Score using letter-based scoring (Qwen3 outputs letter answers)."""
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"({chr(65+i)}) {choice}\n"
    prompt += "Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    scores = _score_logits_by_letter(tokenizer, logits, len(choices))
    
    answer_letter = _get_answer_letter(answer, choices)
    answer_idx = ord(answer_letter) - ord('A')
    
    predicted_idx = np.argmax(scores)
    predicted_choice = choices[predicted_idx]
    
    return {
        "scores": scores,
        "answer_prob": scores[answer_idx],
        "predicted": predicted_choice,
        "correct": predicted_choice == answer
    }


def score_letter_only(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                     question: str, choices: List[str], answer: str) -> Dict:
    """Score using only letter tokens (same as full_text for Qwen3 MCQ)."""
    return score_full_text(model, tokenizer, question, choices, answer)


def score_with_kv(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 question: str, choices: List[str], answer: str,
                 kv_cache) -> Dict:
    """Score with injected KV cache using letter-based scoring."""
    prompt = "Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Convert to DynamicCache if needed
    if isinstance(kv_cache, tuple) and len(kv_cache) == 2:
        k_list, v_list = kv_cache
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(zip(k_list, v_list)):
            cache.update(k, v, layer_idx)
        kv_cache = cache
    
    # Account for cached tokens in position_ids and attention_mask
    seq_len = kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else kv_cache._seen_tokens
    new_len = inputs['input_ids'].shape[1]
    total_len = seq_len + new_len
    
    position_ids = torch.arange(seq_len, total_len, device=model.device).unsqueeze(0)
    attention_mask = torch.ones(1, total_len, device=model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            past_key_values=kv_cache,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits[0, -1, :]
    
    scores = _score_logits_by_letter(tokenizer, logits, len(choices))
    
    answer_letter = _get_answer_letter(answer, choices)
    answer_idx = ord(answer_letter) - ord('A')
    
    predicted_idx = np.argmax(scores)
    predicted_choice = choices[predicted_idx]
    
    return {
        "scores": scores,
        "answer_prob": scores[answer_idx],
        "predicted": predicted_choice,
        "correct": predicted_choice == answer
    }

# ============================================================================
# Comprehensive Experiment Runner
# ============================================================================

class ComprehensiveExperimentRunner:
    """Run comprehensive experiments with all configurations."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.results_dir = Path("experiments/comprehensive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.results_dir / self.experiment_id
        self.output_dir.mkdir(exist_ok=True)
        
        # Save all prompts
        self._save_prompts()
        
        # Record metadata
        self.metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "gpu": self._get_gpu_info(),
            "model_pairs": MODEL_PAIRS,
            "n_diverse_prompts": len(DIVERSE_PROMPTS),
            "n_adversarial_prompts": len(ADVERSARIAL_PROMPTS),
            "param_configs": PARAM_CONFIGS,
        }
        
        self._save_metadata()
    
    def _get_git_hash(self) -> str:
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_gpu_info(self) -> str:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _save_prompts(self):
        """Save all prompts to files."""
        all_prompts = {
            "diverse": DIVERSE_PROMPTS,
            "adversarial": ADVERSARIAL_PROMPTS,
            "total": len(DIVERSE_PROMPTS) + len(ADVERSARIAL_PROMPTS),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "all_prompts.json", 'w') as f:
            json.dump(all_prompts, f, indent=2)
        
        logger.info(f"Saved {all_prompts['total']} prompts to {self.output_dir / 'all_prompts.json'}")
    
    def _save_metadata(self):
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def run_single_config(self, teacher_name: str, student_name: str,
                         prompts: List[Dict], seed: int, 
                         lambda_val: float, mapping_strategy: str,
                         scoring_method: str) -> Dict:
        """Run a single configuration."""
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load models
        teacher_model, teacher_tokenizer = self.model_manager.load_model(teacher_name)
        student_model, student_tokenizer = self.model_manager.load_model(student_name)
        
        results = {
            "config": {
                "teacher": teacher_name,
                "student": student_name,
                "seed": seed,
                "lambda": lambda_val,
                "mapping_strategy": mapping_strategy,
                "scoring_method": scoring_method,
                "n_prompts": len(prompts),
            },
            "scores": {},
            "per_prompt": []
        }
        
        # Get teacher KV caches
        teacher_kv_caches = []
        for prompt in prompts:
            context = f"{prompt['question']}\n"
            for i, choice in enumerate(prompt['choices']):
                context += f"({chr(65+i)}) {choice}\n"
            context += "Answer:"
            
            inputs = teacher_tokenizer(context, return_tensors="pt").to(teacher_model.device)
            with torch.no_grad():
                outputs = teacher_model(**inputs, use_cache=True)
            teacher_kv_caches.append(outputs.past_key_values)
        
        # Student baseline
        student_scores = []
        for prompt in prompts:
            if scoring_method == "full_text":
                score = score_full_text(student_model, student_tokenizer,
                                       prompt['question'], prompt['choices'], prompt['answer'])
            else:
                score = score_letter_only(student_model, student_tokenizer,
                                         prompt['question'], prompt['choices'], prompt['answer'])
            student_scores.append(score['answer_prob'])
        
        results['scores']['student_baseline'] = {
            "mean": np.mean(student_scores),
            "std": np.std(student_scores),
            "scores": student_scores
        }
        
        # Get student head dimension
        student_head_dim = student_model.config.hidden_size // student_model.config.num_attention_heads
        
        # Native average
        native_scores = []
        for i, prompt in enumerate(prompts):
            k_list, v_list = native_average_kv(teacher_kv_caches[i], 
                                          student_model.config.num_hidden_layers,
                                          student_head_dim)
            native_kv = create_kv_cache(k_list, v_list)
            score = score_with_kv(student_model, student_tokenizer,
                                prompt['question'], prompt['choices'], prompt['answer'],
                                native_kv)
            native_scores.append(score['answer_prob'])
        
        results['scores']['native'] = {
            "mean": np.mean(native_scores),
            "std": np.std(native_scores),
            "scores": native_scores,
            "chg": np.mean(native_scores) - np.mean(student_scores)
        }
        
        # Ridge simple
        ridge_scores = []
        for i, prompt in enumerate(prompts):
            k_list, v_list = ridge_simple_kv(teacher_kv_caches[i], 
                                      student_model.config.num_hidden_layers,
                                      lambda_val,
                                      student_head_dim)
            ridge_kv = create_kv_cache(k_list, v_list)
            score = score_with_kv(student_model, student_tokenizer,
                                prompt['question'], prompt['choices'], prompt['answer'],
                                ridge_kv)
            ridge_scores.append(score['answer_prob'])
        
        results['scores']['ridge_simple'] = {
            "mean": np.mean(ridge_scores),
            "std": np.std(ridge_scores),
            "scores": ridge_scores,
            "chg": np.mean(ridge_scores) - np.mean(student_scores)
        }
        
        # Weighted average
        weighted_scores = []
        for i, prompt in enumerate(prompts):
            k_list, v_list = weighted_average_kv(teacher_kv_caches[i], 
                                            student_model.config.num_hidden_layers,
                                            weights=None,
                                            student_head_dim=student_head_dim)
            weighted_kv = create_kv_cache(k_list, v_list)
            score = score_with_kv(student_model, student_tokenizer,
                                prompt['question'], prompt['choices'], prompt['answer'],
                                weighted_kv)
            weighted_scores.append(score['answer_prob'])
        
        results['scores']['weighted_average'] = {
            "mean": np.mean(weighted_scores),
            "std": np.std(weighted_scores),
            "scores": weighted_scores,
            "chg": np.mean(weighted_scores) - np.mean(student_scores)
        }
        
        # Save per-prompt results
        for i, prompt in enumerate(prompts):
            results['per_prompt'].append({
                "prompt_id": prompt['id'],
                "question": prompt['question'],
                "domain": prompt.get('domain', 'unknown'),
                "student_score": student_scores[i],
                "native_score": native_scores[i],
                "ridge_score": ridge_scores[i],
                "weighted_score": weighted_scores[i],
            })
        
        # Unload models
        self.model_manager.unload_model(teacher_name)
        self.model_manager.unload_model(student_name)
        
        return results
    
    def run_all_configs(self) -> Dict:
        """Run all configurations."""
        all_results = {}
        
        # Generate all configurations
        configs = list(product(
            MODEL_PAIRS,
            PARAM_CONFIGS["seeds"],
            PARAM_CONFIGS["lambda_values"],
            PARAM_CONFIGS["mapping_strategies"],
            PARAM_CONFIGS["scoring_methods"]
        ))
        
        logger.info(f"Running {len(configs)} configurations")
        
        for idx, (pair, seed, lambda_val, mapping_strategy, scoring_method) in enumerate(configs):
            teacher, student = pair
            config_key = f"{teacher}→{student}_s{seed}_l{lambda_val}_{mapping_strategy}_{scoring_method}"
            
            logger.info(f"\n[{idx+1}/{len(configs)}] Running {config_key}")
            
            try:
                # Use diverse prompts for this config
                results = self.run_single_config(
                    teacher, student, DIVERSE_PROMPTS, seed,
                    lambda_val, mapping_strategy, scoring_method
                )
                
                all_results[config_key] = results
                
                # Save intermediate results
                with open(self.output_dir / f"{config_key}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error in {config_key}: {e}")
                all_results[config_key] = {"error": str(e)}
        
        # Run adversarial prompts with best config
        logger.info("\nRunning adversarial prompts...")
        for teacher, student in MODEL_PAIRS:
            config_key = f"{teacher}→{student}_adversarial"
            try:
                results = self.run_single_config(
                    teacher, student, ADVERSARIAL_PROMPTS, 42,
                    1e-5, "native_average", "full_text"
                )
                all_results[config_key] = results
                
                with open(self.output_dir / f"{config_key}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error in {config_key}: {e}")
                all_results[config_key] = {"error": str(e)}
        
        # Save all results
        with open(self.output_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def print_summary(self, all_results: Dict):
        """Print comprehensive summary."""
        print("\n" + "="*100)
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*100)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Timestamp: {self.metadata['timestamp']}")
        print(f"GPU: {self.metadata['gpu']}")
        print(f"Total configurations: {len(all_results)}")
        print("="*100)
        
        # Group by model pair
        for teacher, student in MODEL_PAIRS:
            pair_results = {k: v for k, v in all_results.items() 
                          if k.startswith(f"{teacher}→{student}") and 'error' not in v}
            
            if not pair_results:
                continue
            
            print(f"\n{'='*80}")
            print(f"MODEL PAIR: {teacher} → {student}")
            print(f"{'='*80}")
            
            # Aggregate by method
            methods = ['student_baseline', 'native', 'ridge_simple', 'weighted_average']
            for method in methods:
                scores = []
                chgs = []
                for k, v in pair_results.items():
                    if method in v.get('scores', {}):
                        scores.append(v['scores'][method]['mean'])
                        if 'chg' in v['scores'][method]:
                            chgs.append(v['scores'][method]['chg'])
                
                if scores:
                    print(f"\n{method}:")
                    print(f"  Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                    if chgs:
                        print(f"  CHG: {np.mean(chgs):+.4f} ± {np.std(chgs):.4f}")
        
        print("\n" + "="*100)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run comprehensive experiments."""
    logger.info("Starting comprehensive experiment with actual models")
    
    runner = ComprehensiveExperimentRunner()
    all_results = runner.run_all_configs()
    runner.print_summary(all_results)
    
    logger.info("Experiment completed")
    print(f"\nResults saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
