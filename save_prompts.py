#!/usr/bin/env python3
"""Save all prompts to files with metadata."""

import json
import hashlib
from datetime import datetime
from pathlib import Path

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Calibration prompts (30)
calibration_prompts = [
    {"id": "cal_001", "domain": "factual", "question": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Madrid"], "answer": "A"},
    {"id": "cal_002", "domain": "science", "question": "What is the chemical symbol for water?", "choices": ["H2O", "CO2", "NaCl", "O2"], "answer": "A"},
    {"id": "cal_003", "domain": "mathematical", "question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "answer": "B"},
    {"id": "cal_004", "domain": "reasoning", "question": "If all dogs are animals, and all animals need food, then all dogs need food. This is:", "choices": ["Inductive", "Deductive", "Abductive", "Analogical"], "answer": "B"},
    {"id": "cal_005", "domain": "history", "question": "In which year did WWII end?", "choices": ["1943", "1944", "1945", "1946"], "answer": "C"},
    {"id": "cal_006", "domain": "geography", "question": "What is the largest continent?", "choices": ["Africa", "Asia", "Europe", "North America"], "answer": "B"},
    {"id": "cal_007", "domain": "language", "question": "What is a synonym for 'happy'?", "choices": ["Sad", "Joyful", "Angry", "Tired"], "answer": "B"},
    {"id": "cal_008", "domain": "common_sense", "question": "What do you use to write on paper?", "choices": ["Fork", "Pen", "Hammer", "Spoon"], "answer": "B"},
    {"id": "cal_009", "domain": "technical", "question": "What does CPU stand for?", "choices": ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Computer Processing Unit"], "answer": "A"},
    {"id": "cal_010", "domain": "creative", "question": "What color is the sky on a clear day?", "choices": ["Green", "Blue", "Red", "Yellow"], "answer": "B"},
    {"id": "cal_011", "domain": "factual", "question": "What is the largest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "D"},
    {"id": "cal_012", "domain": "science", "question": "What planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": "C"},
    {"id": "cal_013", "domain": "mathematical", "question": "What is 10 - 5?", "choices": ["3", "4", "5", "6"], "answer": "C"},
    {"id": "cal_014", "domain": "reasoning", "question": "If it rains, the ground gets wet. The ground is wet. Therefore:", "choices": ["It rained", "It might have rained", "The ground is always wet", "None of the above"], "answer": "B"},
    {"id": "cal_015", "domain": "history", "question": "Who was the first president of the USA?", "choices": ["Lincoln", "Washington", "Jefferson", "Adams"], "answer": "B"},
    {"id": "cal_016", "domain": "geography", "question": "What is the capital of Japan?", "choices": ["Seoul", "Beijing", "Tokyo", "Bangkok"], "answer": "C"},
    {"id": "cal_017", "domain": "language", "question": "What is the plural of 'child'?", "choices": ["Childs", "Children", "Childes", "Childies"], "answer": "B"},
    {"id": "cal_018", "domain": "common_sense", "question": "What do you use to cut food?", "choices": ["Spoon", "Fork", "Knife", "Plate"], "answer": "C"},
    {"id": "cal_019", "domain": "technical", "question": "What does RAM stand for?", "choices": ["Random Access Memory", "Read Access Memory", "Random Average Memory", "Read Average Memory"], "answer": "A"},
    {"id": "cal_020", "domain": "creative", "question": "What season comes after winter?", "choices": ["Summer", "Fall", "Spring", "Winter"], "answer": "C"},
    {"id": "cal_021", "domain": "factual", "question": "What is the hardest natural substance?", "choices": ["Gold", "Iron", "Diamond", "Silver"], "answer": "C"},
    {"id": "cal_022", "domain": "science", "question": "What gas do plants absorb?", "choices": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"], "answer": "C"},
    {"id": "cal_023", "domain": "mathematical", "question": "What is 3 × 4?", "choices": ["7", "10", "12", "14"], "answer": "C"},
    {"id": "cal_024", "domain": "reasoning", "question": "All birds can fly. A penguin is a bird. Therefore:", "choices": ["Penguins can fly", "Penguins cannot fly", "The premise is wrong", "None of the above"], "answer": "C"},
    {"id": "cal_025", "domain": "history", "question": "In which century did Columbus discover America?", "choices": ["14th", "15th", "16th", "17th"], "answer": "B"},
    {"id": "cal_026", "domain": "geography", "question": "What is the longest river?", "choices": ["Amazon", "Nile", "Mississippi", "Yangtze"], "answer": "B"},
    {"id": "cal_027", "domain": "language", "question": "What is an antonym for 'big'?", "choices": ["Large", "Huge", "Small", "Tall"], "answer": "C"},
    {"id": "cal_028", "domain": "common_sense", "question": "What do you use to open a door?", "choices": ["Key", "Fork", "Pen", "Spoon"], "answer": "A"},
    {"id": "cal_029", "domain": "technical", "question": "What does HTML stand for?", "choices": ["Hyper Text Markup Language", "High Tech Modern Language", "Home Tool Markup Language", "Hyperlink Text Mode Language"], "answer": "A"},
    {"id": "cal_030", "domain": "creative", "question": "What is the opposite of 'hot'?", "choices": ["Warm", "Cold", "Cool", "Mild"], "answer": "B"},
]

# Held-out prompts (20)
heldout_prompts = [
    {"id": "val_001", "domain": "factual", "question": "What is the capital of Germany?", "choices": ["Munich", "Berlin", "Hamburg", "Frankfurt"], "answer": "B"},
    {"id": "val_002", "domain": "science", "question": "What is the boiling point of water?", "choices": ["90°C", "100°C", "110°C", "120°C"], "answer": "B"},
    {"id": "val_003", "domain": "mathematical", "question": "What is 15 ÷ 3?", "choices": ["3", "4", "5", "6"], "answer": "C"},
    {"id": "val_004", "domain": "reasoning", "question": "If A > B and B > C, then:", "choices": ["A > C", "A < C", "A = C", "Cannot determine"], "answer": "A"},
    {"id": "val_005", "domain": "history", "question": "When did the French Revolution begin?", "choices": ["1776", "1789", "1799", "1804"], "answer": "B"},
    {"id": "val_006", "domain": "geography", "question": "What is the smallest country?", "choices": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"], "answer": "B"},
    {"id": "val_007", "domain": "language", "question": "What is a verb?", "choices": ["Happy", "Run", "Beautiful", "House"], "answer": "B"},
    {"id": "val_008", "domain": "common_sense", "question": "What do you use to see?", "choices": ["Ears", "Eyes", "Nose", "Mouth"], "answer": "B"},
    {"id": "val_009", "domain": "technical", "question": "What does GPU stand for?", "choices": ["Graphics Processing Unit", "General Processing Unit", "Graphics Program Utility", "General Purpose Unit"], "answer": "A"},
    {"id": "val_010", "domain": "creative", "question": "What is the opposite of 'dark'?", "choices": ["Dim", "Bright", "Shadow", "Night"], "answer": "B"},
    {"id": "val_011", "domain": "factual", "question": "What is the largest planet?", "choices": ["Saturn", "Jupiter", "Neptune", "Uranus"], "answer": "B"},
    {"id": "val_012", "domain": "science", "question": "What is the speed of light?", "choices": ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "answer": "A"},
    {"id": "val_013", "domain": "mathematical", "question": "What is √16?", "choices": ["2", "3", "4", "5"], "answer": "C"},
    {"id": "val_014", "domain": "reasoning", "question": "If no A are B, and all C are A, then:", "choices": ["All C are B", "No C are B", "Some C are B", "Cannot determine"], "answer": "B"},
    {"id": "val_015", "domain": "history", "question": "Who wrote the Declaration of Independence?", "choices": ["Washington", "Jefferson", "Franklin", "Adams"], "answer": "B"},
    {"id": "val_016", "domain": "geography", "question": "What is the deepest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "D"},
    {"id": "val_017", "domain": "language", "question": "What is a noun?", "choices": ["Run", "Happy", "Dog", "Quickly"], "answer": "C"},
    {"id": "val_018", "domain": "common_sense", "question": "What do you use to listen to music?", "choices": ["Eyes", "Ears", "Nose", "Hands"], "answer": "B"},
    {"id": "val_019", "domain": "technical", "question": "What does USB stand for?", "choices": ["Universal Serial Bus", "United Serial Bus", "Universal System Bus", "United System Bus"], "answer": "A"},
    {"id": "val_020", "domain": "creative", "question": "What is the opposite of 'fast'?", "choices": ["Quick", "Rapid", "Slow", "Speedy"], "answer": "C"},
]

def compute_hash(data):
    """Compute SHA256 hash."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

# Save calibration prompts
cal_data = {
    "split": "calibration",
    "n_prompts": len(calibration_prompts),
    "timestamp": datetime.now().isoformat(),
    "hash": compute_hash(calibration_prompts),
    "prompts": calibration_prompts
}

with open(data_dir / "calibration_prompts.json", 'w') as f:
    json.dump(cal_data, f, indent=2)

# Save held-out prompts
val_data = {
    "split": "heldout",
    "n_prompts": len(heldout_prompts),
    "timestamp": datetime.now().isoformat(),
    "hash": compute_hash(heldout_prompts),
    "prompts": heldout_prompts
}

with open(data_dir / "heldout_prompts.json", 'w') as f:
    json.dump(val_data, f, indent=2)

# Save all prompts
all_prompts = calibration_prompts + heldout_prompts
all_data = {
    "split": "all",
    "n_prompts": len(all_prompts),
    "timestamp": datetime.now().isoformat(),
    "hash": compute_hash(all_prompts),
    "calibration_hash": compute_hash(calibration_prompts),
    "heldout_hash": compute_hash(heldout_prompts),
    "prompts": all_prompts
}

with open(data_dir / "all_prompts.json", 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"Saved {len(calibration_prompts)} calibration prompts to {data_dir / 'calibration_prompts.json'}")
print(f"Saved {len(heldout_prompts)} held-out prompts to {data_dir / 'heldout_prompts.json'}")
print(f"Saved {len(all_prompts)} total prompts to {data_dir / 'all_prompts.json'}")
print(f"\nHashes:")
print(f"  Calibration: {compute_hash(calibration_prompts)}")
print(f"  Held-out: {compute_hash(heldout_prompts)}")
print(f"  All: {compute_hash(all_prompts)}")
