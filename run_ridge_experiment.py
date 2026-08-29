#!/usr/bin/env python3
"""Cross-Model KV State Handoff Experiment — Correct Pipeline.

Uses the existing codebase's RidgePerHeadMapper with fit_ridge_aggregate for proper
Gram matrix accumulation across calibration samples.
"""

import json
import os
import gc
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_BASE = os.path.expanduser("~/.cache/modelscope/models/Qwen--Qwen3-{size}/snapshots/master")
MODEL_SIZES = ["0.6B", "1.7B", "4B", "8B"]

MODEL_CONFIGS = {
    "0.6B": {"layers": 28, "hidden": 1024, "heads": 16, "kv_heads": 8, "head_dim": 128},
    "1.7B": {"layers": 28, "hidden": 2048, "heads": 16, "kv_heads": 8, "head_dim": 128},
    "4B":   {"layers": 36, "hidden": 2560, "heads": 32, "kv_heads": 8, "head_dim": 128},
    "8B":   {"layers": 36, "hidden": 4096, "heads": 32, "kv_heads": 8, "head_dim": 128},
}

PAIRS = [
    ("4B", "1.7B"),
    ("4B", "0.6B"),
    ("8B", "1.7B"),
    ("8B", "0.6B"),
]

# ============================================================================
# MCQ Prompts (62 total: 50 diverse + 12 adversarial)
# ============================================================================

DIVERSE_PROMPTS = [
    {"id": "math_01", "domain": "math", "question": "What is 15 + 27?", "choices": ["42", "43", "41", "44"], "answer": "42"},
    {"id": "math_02", "domain": "math", "question": "What is 8 × 7?", "choices": ["54", "56", "58", "63"], "answer": "56"},
    {"id": "math_03", "domain": "math", "question": "What is 144 ÷ 12?", "choices": ["11", "12", "13", "14"], "answer": "12"},
    {"id": "math_04", "domain": "math", "question": "What is the square root of 81?", "choices": ["7", "8", "9", "10"], "answer": "9"},
    {"id": "math_05", "domain": "math", "question": "What is 2^10?", "choices": ["512", "1024", "2048", "256"], "answer": "1024"},
    {"id": "sci_01", "domain": "science", "question": "What planet is closest to the Sun?", "choices": ["Venus", "Mercury", "Mars", "Earth"], "answer": "Mercury"},
    {"id": "sci_02", "domain": "science", "question": "What is the chemical symbol for water?", "choices": ["HO", "H2O", "OH2", "H3O"], "answer": "H2O"},
    {"id": "sci_03", "domain": "science", "question": "How many bones are in the adult human body?", "choices": ["186", "206", "216", "256"], "answer": "206"},
    {"id": "sci_04", "domain": "science", "question": "What gas do plants absorb from the atmosphere?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "answer": "Carbon dioxide"},
    {"id": "sci_05", "domain": "science", "question": "What is the speed of light approximately?", "choices": ["300,000 km/s", "150,000 km/s", "500,000 km/s", "100,000 km/s"], "answer": "300,000 km/s"},
    {"id": "geo_01", "domain": "geography", "question": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "answer": "Paris"},
    {"id": "geo_02", "domain": "geography", "question": "Which continent is Egypt in?", "choices": ["Asia", "Europe", "Africa", "South America"], "answer": "Africa"},
    {"id": "geo_03", "domain": "geography", "question": "What is the longest river in the world?", "choices": ["Amazon", "Nile", "Mississippi", "Yangtze"], "answer": "Nile"},
    {"id": "geo_04", "domain": "geography", "question": "What country has the most people?", "choices": ["United States", "India", "China", "Indonesia"], "answer": "India"},
    {"id": "geo_05", "domain": "geography", "question": "What is the largest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "Pacific"},
    {"id": "hist_01", "domain": "history", "question": "In what year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": "1945"},
    {"id": "hist_02", "domain": "history", "question": "Who was the first President of the United States?", "choices": ["Jefferson", "Adams", "Washington", "Franklin"], "answer": "Washington"},
    {"id": "hist_03", "domain": "history", "question": "In what year did the Berlin Wall fall?", "choices": ["1987", "1988", "1989", "1990"], "answer": "1989"},
    {"id": "hist_04", "domain": "history", "question": "Which ancient civilization built the pyramids?", "choices": ["Roman", "Greek", "Egyptian", "Mayan"], "answer": "Egyptian"},
    {"id": "hist_05", "domain": "history", "question": "What year was the Declaration of Independence signed?", "choices": ["1774", "1775", "1776", "1777"], "answer": "1776"},
    {"id": "lit_01", "domain": "literature", "question": "Who wrote 'Romeo and Juliet'?", "choices": ["Dickens", "Shakespeare", "Austen", "Twain"], "answer": "Shakespeare"},
    {"id": "lit_02", "domain": "literature", "question": "What is the first book of the Bible?", "choices": ["Exodus", "Leviticus", "Genesis", "Numbers"], "answer": "Genesis"},
    {"id": "lit_03", "domain": "literature", "question": "Who wrote '1984'?", "choices": ["Huxley", "Orwell", "Bradbury", "Vonnegut"], "answer": "Orwell"},
    {"id": "lit_04", "domain": "literature", "question": "In 'The Great Gatsby', what color is the light at the end of Daisy's dock?", "choices": ["Red", "Green", "Blue", "White"], "answer": "Green"},
    {"id": "lit_05", "domain": "literature", "question": "Who wrote 'Pride and Prejudice'?", "choices": ["Bronte", "Austen", "Eliot", "Woolf"], "answer": "Austen"},
    {"id": "tech_01", "domain": "technology", "question": "What does CPU stand for?", "choices": ["Central Processing Unit", "Computer Processing Unit", "Central Program Unit", "Core Processing Unit"], "answer": "Central Processing Unit"},
    {"id": "tech_02", "domain": "technology", "question": "Who co-founded Apple with Steve Jobs?", "choices": ["Bill Gates", "Steve Wozniak", "Paul Allen", "Michael Dell"], "answer": "Steve Wozniak"},
    {"id": "tech_03", "domain": "technology", "question": "What does HTML stand for?", "choices": ["Hyper Text Markup Language", "High Tech Modern Language", "Hyper Transfer Markup Language", "Home Tool Markup Language"], "answer": "Hyper Text Markup Language"},
    {"id": "tech_04", "domain": "technology", "question": "In what year was the first iPhone released?", "choices": ["2005", "2006", "2007", "2008"], "answer": "2007"},
    {"id": "tech_05", "domain": "technology", "question": "What programming language was created by Guido van Rossum?", "choices": ["Java", "C++", "Python", "Ruby"], "answer": "Python"},
    {"id": "music_01", "domain": "music", "question": "How many strings does a standard guitar have?", "choices": ["4", "5", "6", "7"], "answer": "6"},
    {"id": "music_02", "domain": "music", "question": "Who composed 'The Four Seasons'?", "choices": ["Mozart", "Bach", "Vivaldi", "Beethoven"], "answer": "Vivaldi"},
    {"id": "music_03", "domain": "music", "question": "What instrument has 88 keys?", "choices": ["Organ", "Piano", "Harpsichord", "Accordion"], "answer": "Piano"},
    {"id": "music_04", "domain": "music", "question": "How many musicians are in a standard quartet?", "choices": ["2", "3", "4", "5"], "answer": "4"},
    {"id": "music_05", "domain": "music", "question": "What does 'forte' mean in music?", "choices": ["Slow", "Fast", "Loud", "Soft"], "answer": "Loud"},
    {"id": "sport_01", "domain": "sports", "question": "How many players are on a basketball team on the court?", "choices": ["4", "5", "6", "7"], "answer": "5"},
    {"id": "sport_02", "domain": "sports", "question": "In which sport is the term 'love' used?", "choices": ["Golf", "Tennis", "Cricket", "Badminton"], "answer": "Tennis"},
    {"id": "sport_03", "domain": "sports", "question": "How many holes are in a standard round of golf?", "choices": ["9", "18", "27", "36"], "answer": "18"},
    {"id": "sport_04", "domain": "sports", "question": "What country invented baseball?", "choices": ["England", "United States", "Japan", "Canada"], "answer": "United States"},
    {"id": "sport_05", "domain": "sports", "question": "How long is an Olympic swimming pool?", "choices": ["25 meters", "50 meters", "75 meters", "100 meters"], "answer": "50 meters"},
    {"id": "phil_01", "domain": "philosophy", "question": "Who said 'I think, therefore I am'?", "choices": ["Plato", "Aristotle", "Descartes", "Kant"], "answer": "Descartes"},
    {"id": "phil_02", "domain": "philosophy", "question": "What is the study of knowledge called?", "choices": ["Ethics", "Logic", "Epistemology", "Metaphysics"], "answer": "Epistemology"},
    {"id": "phil_03", "domain": "philosophy", "question": "Who wrote 'The Republic'?", "choices": ["Aristotle", "Plato", "Socrates", "Epicurus"], "answer": "Plato"},
    {"id": "phil_04", "domain": "philosophy", "question": "What is the categorical imperative associated with?", "choices": ["Hegel", "Nietzsche", "Kant", "Locke"], "answer": "Kant"},
    {"id": "phil_05", "domain": "philosophy", "question": "Who is known as the father of modern philosophy?", "choices": ["Kant", "Descartes", "Hume", "Leibniz"], "answer": "Descartes"},
    {"id": "lang_01", "domain": "language", "question": "What is the most spoken language in the world?", "choices": ["English", "Spanish", "Mandarin Chinese", "Hindi"], "answer": "Mandarin Chinese"},
    {"id": "lang_02", "domain": "language", "question": "How many official languages does the UN have?", "choices": ["4", "5", "6", "7"], "answer": "6"},
    {"id": "lang_03", "domain": "language", "question": "What language has the most native speakers?", "choices": ["English", "Spanish", "Hindi", "Mandarin Chinese"], "answer": "Mandarin Chinese"},
    {"id": "lang_04", "domain": "language", "question": "What is the alphabet with the fewest letters?", "choices": ["Japanese", "Hawaiian", "Rotokas", "Piraha"], "answer": "Rotokas"},
    {"id": "lang_05", "domain": "language", "question": "What language uses the Cyrillic alphabet?", "choices": ["Greek", "Arabic", "Russian", "Hebrew"], "answer": "Russian"},
]

ADVERSARIAL_PROMPTS = [
    {"id": "adv_01", "domain": "adversarial", "question": "If a hen and a half lays an egg and a half in a day and a half, how many eggs does one hen lay in one day?", "choices": ["1", "1.5", "2", "0.5"], "answer": "1"},
    {"id": "adv_02", "domain": "adversarial", "question": "What has keys but no locks?", "choices": ["Door", "Piano", "Map", "Keyboard"], "answer": "Piano"},
    {"id": "adv_03", "domain": "adversarial", "question": "I am taken from a mine, and shut up in a wooden case, from which I am never released, and yet I am used by almost every person. What am I?", "choices": ["Gold", "Coal", "Pencil lead", "Diamond"], "answer": "Pencil lead"},
    {"id": "adv_04", "domain": "adversarial", "question": "What gets wetter the more it dries?", "choices": ["Sponge", "Towel", "Sun", "River"], "answer": "Towel"},
    {"id": "adv_05", "domain": "adversarial", "question": "What can travel around the world while staying in a corner?", "choices": ["Airplane", "Stamp", "Satellite", "Internet"], "answer": "Stamp"},
    {"id": "adv_06", "domain": "adversarial", "question": "The more you take, the more you leave behind. What am I?", "choices": ["Footsteps", "Photos", "Memories", "Breaths"], "answer": "Footsteps"},
    {"id": "adv_07", "domain": "adversarial", "question": "What is always in front of you but can't be seen?", "choices": ["Future", "Air", "Nose", "Wind"], "answer": "Future"},
    {"id": "adv_08", "domain": "adversarial", "question": "What has a head and a tail but no body?", "choices": ["Snake", "Coin", "Arrow", "Comet"], "answer": "Coin"},
    {"id": "adv_09", "domain": "adversarial", "question": "What comes once in a minute, twice in a moment, but never in a thousand years?", "choices": ["Time", "The letter m", "Opportunity", "Silence"], "answer": "The letter m"},
    {"id": "adv_10", "domain": "adversarial", "question": "What is seen in the middle of March and April that can't be seen at the beginning or end of either month?", "choices": ["Spring", "The letter r", "Easter", "Rain"], "answer": "The letter r"},
    {"id": "adv_11", "domain": "adversarial", "question": "What 5-letter word becomes shorter when you add 2 letters to it?", "choices": ["Small", "Short", "Tiny", "Brief"], "answer": "Short"},
    {"id": "adv_12", "domain": "adversarial", "question": "What disappears as soon as you say its name?", "choices": ["Secret", "Silence", "Nothing", "Shadow"], "answer": "Silence"},
]

ALL_PROMPTS = DIVERSE_PROMPTS + ADVERSARIAL_PROMPTS

# ============================================================================
# Scoring Functions
# ============================================================================

def score_by_letters(tokenizer, logits: torch.Tensor, n_choices: int = 4) -> List[float]:
    """Score logits by letter tokens (A/B/C/D), normalized."""
    probs = torch.softmax(logits, dim=-1)
    scores = []
    for i in range(n_choices):
        letter = chr(65 + i)
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        token_id = ids[-1] if ids else tokenizer.encode(letter, add_special_tokens=False)[0]
        scores.append(probs[token_id].item())
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    return scores


def get_answer_idx(answer: str, choices: List[str]) -> int:
    """Get letter index (0=A, 1=B, etc.) for the correct answer."""
    return choices.index(answer)


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format MCQ prompt."""
    prompt = f"{question}\n"
    for i, c in enumerate(choices):
        prompt += f"({chr(65+i)}) {c}\n"
    prompt += "Answer:"
    return prompt


# ============================================================================
# Model Management
# ============================================================================

class ModelManager:
    """Load/unload models to manage GPU memory."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load(self, size: str):
        """Load model and tokenizer."""
        if size in self.models:
            return self.models[size], self.tokenizers[size]
        
        path = MODEL_BASE.format(size=size)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.models[size] = model
        self.tokenizers[size] = tokenizer
        return model, tokenizer
    
    def unload(self, size: str):
        """Unload model to free GPU memory."""
        if size in self.models:
            del self.models[size]
            del self.tokenizers[size]
            gc.collect()
            torch.cuda.empty_cache()


# ============================================================================
# KV Cache Extraction and Injection
# ============================================================================

def extract_kv_cache(model, tokenizer, text: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run model on text and extract K and V as numpy arrays.
    
    Returns:
        k: (L, S, H, D) numpy array (float32)
        v: (L, S, H, D) numpy array (float32)
        seq_len: sequence length
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    
    kv = out.past_key_values
    seq_len = kv._seen_tokens
    
    # Stack layers: each key_cache[i] is (1, H, S, D)
    # Convert to float32 for numpy compatibility
    k_list = [kv.key_cache[i].squeeze(0).cpu().float().numpy() for i in range(len(kv.key_cache))]
    v_list = [kv.value_cache[i].squeeze(0).cpu().float().numpy() for i in range(len(kv.value_cache))]
    
    # Stack to (L, S, H, D)
    k = np.stack(k_list, axis=0)
    v = np.stack(v_list, axis=0)
    
    return k, v, seq_len


def create_dynamic_cache_from_numpy(k: np.ndarray, v: np.ndarray) -> object:
    """Create DynamicCache from numpy arrays."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    L = k.shape[0]
    for i in range(L):
        k_tensor = torch.from_numpy(k[i]).unsqueeze(0).to(torch.float16)  # (1, H, S, D)
        v_tensor = torch.from_numpy(v[i]).unsqueeze(0).to(torch.float16)
        cache.update(k_tensor, v_tensor, i)
    return cache


# ============================================================================
# Ridge Mapper Fitting
# ============================================================================

def fit_ridge_mapper(
    calib_teacher_k: List[np.ndarray],  # List of (L_t, S, H, D) arrays
    calib_teacher_v: List[np.ndarray],
    calib_student_k: List[np.ndarray],  # List of (L_s, S, H, D) arrays
    calib_student_v: List[np.ndarray],
    n_teacher_layers: int,
    n_student_layers: int,
    head_dim: int,
    lam: float = 1e-3,
):
    """Fit Ridge per-head mapper using paired calibration data with Gram aggregation.
    
    Args:
        calib_teacher_k: List of teacher K arrays, each (L_t, S, H, D)
        calib_teacher_v: List of teacher V arrays
        calib_student_k: List of student K arrays, each (L_s, S, H, D)
        calib_student_v: List of student V arrays
        n_teacher_layers: Number of teacher layers
        n_student_layers: Number of student layers
        head_dim: Head dimension (128 for all Qwen3)
        lam: Ridge regularization coefficient
    
    Returns:
        mapper_k: Fitted RidgePerHeadMapper for K
        mapper_v: Fitted RidgePerHeadMapper for V
        layer_map: Layer alignment mapping
    """
    from apcs.mapper.math import RidgePerHeadMapper
    from apcs.alignment.runner import proportional_mapping
    from apcs.mapper.aggregate import fit_ridge_aggregate
    
    # Create layer mapping
    layer_map = proportional_mapping(n_teacher_layers, n_student_layers)
    
    # Initialize mappers
    mapper_k = RidgePerHeadMapper(lam=lam)
    mapper_v = RidgePerHeadMapper(lam=lam)
    
    # Fit K mapper using Gram aggregation
    samples_k = [(t_k, s_k) for t_k, s_k in zip(calib_teacher_k, calib_student_k)]
    fit_ridge_aggregate(mapper_k, samples_k, layer_map, kv_kind="K")
    
    # Fit V mapper using Gram aggregation
    samples_v = [(t_v, s_v) for t_v, s_v in zip(calib_teacher_v, calib_student_v)]
    fit_ridge_aggregate(mapper_v, samples_v, layer_map, kv_kind="V")
    
    return mapper_k, mapper_v, layer_map


def transform_kv_with_mapper(
    teacher_k: np.ndarray,  # (L_t, S, H, D)
    teacher_v: np.ndarray,
    mapper_k,  # Fitted RidgePerHeadMapper
    mapper_v,
    layer_map,
):
    """Transform teacher KV to student space using fitted mappers."""
    
    # Transform K
    student_k = mapper_k.transform(teacher_k, layer_map, kv_kind="K")
    
    # Transform V
    student_v = mapper_v.transform(teacher_v, layer_map, kv_kind="V")
    
    return student_k, student_v


# ============================================================================
# Evaluation Protocol
# ============================================================================

def evaluate_student_self(model, tokenizer, prompt_data: Dict) -> Dict:
    """Method 1: Student reads full context itself."""
    question = prompt_data['question']
    choices = prompt_data['choices']
    answer = prompt_data['answer']
    
    text = format_mcq_prompt(question, choices)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    scores = score_by_letters(tokenizer, logits, len(choices))
    pred_idx = int(np.argmax(scores))
    answer_idx = get_answer_idx(answer, choices)
    
    return {
        "method": "student_self",
        "scores": scores,
        "predicted": choices[pred_idx],
        "correct": pred_idx == answer_idx,
        "answer_prob": scores[answer_idx],
    }


def evaluate_teacher_full(model, tokenizer, prompt_data: Dict) -> Dict:
    """Method 2: Teacher reads full context (upper bound)."""
    question = prompt_data['question']
    choices = prompt_data['choices']
    answer = prompt_data['answer']
    
    text = format_mcq_prompt(question, choices)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    scores = score_by_letters(tokenizer, logits, len(choices))
    pred_idx = int(np.argmax(scores))
    answer_idx = get_answer_idx(answer, choices)
    
    return {
        "method": "teacher_full",
        "scores": scores,
        "predicted": choices[pred_idx],
        "correct": pred_idx == answer_idx,
        "answer_prob": scores[answer_idx],
    }


def evaluate_ridge_handoff(
    teacher_model, teacher_tokenizer,
    student_model, student_tokenizer,
    mapper_k, mapper_v, layer_map,
    prompt_data: Dict,
) -> Dict:
    """Method 4: Teacher prefill context → map KV → inject into student."""
    question = prompt_data['question']
    choices = prompt_data['choices']
    answer = prompt_data['answer']
    
    # Teacher processes context only (not the query part)
    context = f"{question}\n"
    for i, c in enumerate(choices):
        context += f"({chr(65+i)}) {c}\n"
    
    # Extract teacher KV
    teacher_k, teacher_v, seq_len = extract_kv_cache(teacher_model, teacher_tokenizer, context)
    
    # Transform KV using fitted mappers
    student_k, student_v = transform_kv_with_mapper(teacher_k, teacher_v, mapper_k, mapper_v, layer_map)
    
    # Create DynamicCache from transformed KV
    cache = create_dynamic_cache_from_numpy(student_k, student_v)
    cache._seen_tokens = seq_len
    
    # Student processes only "Answer:" with injected cache
    answer_token = student_tokenizer.encode("Answer:", add_special_tokens=False)
    answer_inputs = torch.tensor([answer_token], device=student_model.device)
    
    total_len = seq_len + len(answer_token)
    position_ids = torch.arange(seq_len, total_len, device=student_model.device).unsqueeze(0)
    attention_mask = torch.ones(1, total_len, device=student_model.device)
    
    with torch.no_grad():
        logits = student_model(
            input_ids=answer_inputs,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).logits[0, -1, :]
    
    scores = score_by_letters(student_tokenizer, logits, len(choices))
    pred_idx = int(np.argmax(scores))
    answer_idx = get_answer_idx(answer, choices)
    
    return {
        "method": "ridge_handoff",
        "scores": scores,
        "predicted": choices[pred_idx],
        "correct": pred_idx == answer_idx,
        "answer_prob": scores[answer_idx],
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment():
    """Run all experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/ridge_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU info
    try:
        import subprocess
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
    except:
        gpu_info = "unknown"
    
    # Configuration
    n_calib = 20  # Calibration samples
    n_eval = 20   # Evaluation samples
    seeds = [42, 123, 456]
    lambdas = [1e-4, 1e-3, 1e-2]
    
    metadata = {
        "timestamp": timestamp,
        "gpu": gpu_info,
        "n_calib": n_calib,
        "n_eval": n_eval,
        "seeds": seeds,
        "lambdas": lambdas,
        "model_configs": MODEL_CONFIGS,
        "pairs": PAIRS,
        "n_prompts_total": len(ALL_PROMPTS),
    }
    
    all_results = []
    
    # Split prompts: first n_calib for calibration, next n_eval for evaluation
    calib_prompts = ALL_PROMPTS[:n_calib]
    eval_prompts = ALL_PROMPTS[n_calib:n_calib + n_eval]
    
    for teacher_size, student_size in PAIRS:
        teacher_cfg = MODEL_CONFIGS[teacher_size]
        student_cfg = MODEL_CONFIGS[student_size]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {teacher_size} -> {student_size}")
        logger.info(f"Teacher: {teacher_cfg['layers']}L, Student: {student_cfg['layers']}L")
        logger.info(f"{'='*60}")
        
        manager = ModelManager()
        
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            for lam in lambdas:
                logger.info(f"\nSeed={seed}, λ={lam}")
                
                # Load teacher and extract calibration KV
                logger.info("Loading teacher and extracting calibration KV...")
                teacher_model, teacher_tokenizer = manager.load(teacher_size)
                
                calib_teacher_k = []
                calib_teacher_v = []
                for p in calib_prompts:
                    context = format_mcq_prompt(p['question'], p['choices'])
                    k, v, _ = extract_kv_cache(teacher_model, teacher_tokenizer, context)
                    calib_teacher_k.append(k)
                    calib_teacher_v.append(v)
                
                # Unload teacher
                manager.unload(teacher_size)
                
                # Load student and extract calibration KV (for paired calibration)
                logger.info("Loading student and extracting calibration KV...")
                student_model, student_tokenizer = manager.load(student_size)
                
                calib_student_k = []
                calib_student_v = []
                for p in calib_prompts:
                    context = format_mcq_prompt(p['question'], p['choices'])
                    k, v, _ = extract_kv_cache(student_model, student_tokenizer, context)
                    calib_student_k.append(k)
                    calib_student_v.append(v)
                
                # Fit Ridge mapper
                logger.info("Fitting Ridge mapper...")
                mapper_k, mapper_v, layer_map = fit_ridge_mapper(
                    calib_teacher_k, calib_teacher_v,
                    calib_student_k, calib_student_v,
                    teacher_cfg['layers'], student_cfg['layers'],
                    teacher_cfg['head_dim'], lam
                )
                
                # Student baseline (no injection)
                logger.info("Evaluating student baseline...")
                baseline_results = []
                for p in eval_prompts:
                    result = evaluate_student_self(student_model, student_tokenizer, p)
                    baseline_results.append(result)
                
                baseline_acc = sum(r['correct'] for r in baseline_results) / len(baseline_results)
                
                # Ridge handoff (with mapped KV injection)
                logger.info("Evaluating ridge handoff...")
                handoff_results = []
                for p in eval_prompts:
                    # Reload teacher for handoff evaluation
                    teacher_model, teacher_tokenizer = manager.load(teacher_size)
                    
                    result = evaluate_ridge_handoff(
                        teacher_model, teacher_tokenizer,
                        student_model, student_tokenizer,
                        mapper_k, mapper_v, layer_map,
                        p
                    )
                    handoff_results.append(result)
                
                handoff_acc = sum(r['correct'] for r in handoff_results) / len(handoff_results)
                
                # Teacher upper bound
                logger.info("Evaluating teacher upper bound...")
                teacher_model, teacher_tokenizer = manager.load(teacher_size)
                teacher_results = []
                for p in eval_prompts:
                    result = evaluate_teacher_full(teacher_model, teacher_tokenizer, p)
                    teacher_results.append(result)
                
                teacher_acc = sum(r['correct'] for r in teacher_results) / len(teacher_results)
                
                # Record results
                result_entry = {
                    "teacher": teacher_size,
                    "student": student_size,
                    "seed": seed,
                    "lambda": lam,
                    "baseline_accuracy": baseline_acc,
                    "handoff_accuracy": handoff_acc,
                    "teacher_accuracy": teacher_acc,
                    "chg": handoff_acc - baseline_acc,
                    "n_eval": len(eval_prompts),
                    "n_correct_baseline": sum(r['correct'] for r in baseline_results),
                    "n_correct_handoff": sum(r['correct'] for r in handoff_results),
                    "n_correct_teacher": sum(r['correct'] for r in teacher_results),
                }
                all_results.append(result_entry)
                
                logger.info(f"  Baseline: {baseline_acc:.4f}")
                logger.info(f"  Handoff:  {handoff_acc:.4f}")
                logger.info(f"  Teacher:  {teacher_acc:.4f}")
                logger.info(f"  CHG:      {handoff_acc - baseline_acc:+.4f}")
                
                # Unload teacher
                manager.unload(teacher_size)
        
        # Unload student
        manager.unload(student_size)
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump({"metadata": metadata, "results": all_results}, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("EXPERIMENT SUMMARY")
    print("="*90)
    print(f"{'Pair':<15} {'Seed':<8} {'λ':<12} {'Baseline':<10} {'Handoff':<10} {'Teacher':<10} {'CHG':<10}")
    print("-"*90)
    for r in all_results:
        print(f"{r['teacher']}→{r['student']:<10} {r['seed']:<8} {r['lambda']:<12.0e} "
              f"{r['baseline_accuracy']:<10.4f} {r['handoff_accuracy']:<10.4f} "
              f"{r['teacher_accuracy']:<10.4f} {r['chg']:<+10.4f}")
    
    return all_results


if __name__ == "__main__":
    run_experiment()
