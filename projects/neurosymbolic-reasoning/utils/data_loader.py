"""
Data loading utilities for scientific reasoning datasets.

Supports:
- SciQ: Science question answering dataset
- ARC: AI2 Reasoning Challenge
- Synthetic: Generated logical reasoning data

References:
- Welbl et al. (2017): "Crowdsourcing Multiple Choice Science Questions" (SciQ)
- Clark et al. (2018): "Think you have Solved Question Answering?" (ARC)
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ReasoningExample:
    """Single reasoning example."""
    question: str
    choices: List[str]
    correct_answer: int
    support_text: str = ""
    question_id: str = ""
    difficulty: str = "medium"


class SimpleTokenizer:
    """Simple tokenizer fallback when transformers unavailable."""
    
    def __init__(self, vocab_size: int = 30522, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.pad_token_id = 0
    
    def __call__(
        self,
        text: str,
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = None
    ) -> Dict:
        max_len = max_length or self.max_length
        
        # Simple word-level tokenization
        words = text.lower().split()
        
        # Build vocabulary on the fly (deterministic hash)
        tokens = [2]  # [CLS]
        for word in words[:max_len - 2]:
            if word not in self.word_to_idx:
                # Deterministic hash to vocab index
                idx = hash(word) % (self.vocab_size - 4) + 4
                self.word_to_idx[word] = idx
            tokens.append(self.word_to_idx[word])
        tokens.append(3)  # [SEP]
        
        # Pad or truncate
        attention_mask = [1] * len(tokens)
        
        if len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens.extend([0] * pad_len)
            attention_mask.extend([0] * pad_len)
        else:
            tokens = tokens[:max_len]
            attention_mask = attention_mask[:max_len]
        
        result = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result


def get_tokenizer(model_name: str = "bert-base-uncased"):
    """Get tokenizer, falling back to simple tokenizer if needed."""
    if HAS_TRANSFORMERS:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pass
    return SimpleTokenizer()


class SciQDataset(Dataset):
    """
    SciQ dataset for science question answering.
    
    Format: Multiple choice questions with 4 options and support text.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.examples: List[ReasoningExample] = []
        
        self._load_data(max_samples)
    
    def _load_data(self, max_samples: Optional[int] = None):
        """Load and parse SciQ data."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break
                
                example = ReasoningExample(
                    question=item.get("question", ""),
                    choices=[
                        item.get("distractor1", ""),
                        item.get("distractor2", ""),
                        item.get("distractor3", ""),
                        item.get("correct_answer", "")
                    ],
                    correct_answer=3,  # Correct answer is always last in this format
                    support_text=item.get("support", ""),
                    question_id=str(i)
                )
                self.examples.append(example)
        else:
            # Generate synthetic data if file doesn't exist
            self.examples = self._generate_synthetic_data(max_samples or 100)
    
    def _generate_synthetic_data(self, num_samples: int) -> List[ReasoningExample]:
        """Generate synthetic scientific reasoning examples."""
        templates = [
            {
                "question": "What causes {phenomenon}?",
                "correct": "{cause}",
                "distractors": ["{d1}", "{d2}", "{d3}"],
                "support": "{cause} is known to cause {phenomenon}."
            },
            {
                "question": "What is the main function of {entity}?",
                "correct": "{function}",
                "distractors": ["{d1}", "{d2}", "{d3}"],
                "support": "The {entity} is responsible for {function}."
            },
            {
                "question": "Which process results in {outcome}?",
                "correct": "{process}",
                "distractors": ["{d1}", "{d2}", "{d3}"],
                "support": "{process} leads to {outcome}."
            }
        ]
        
        # Scientific domain vocabulary
        science_vocab = {
            "phenomenon": ["photosynthesis", "evaporation", "oxidation", "mitosis", "erosion"],
            "cause": ["sunlight", "heat energy", "oxygen", "cell division", "water flow"],
            "entity": ["mitochondria", "chloroplast", "nucleus", "membrane", "ribosome"],
            "function": ["energy production", "photosynthesis", "genetic storage", "protection", "protein synthesis"],
            "process": ["condensation", "respiration", "fermentation", "diffusion", "osmosis"],
            "outcome": ["water formation", "ATP production", "alcohol", "equilibrium", "cell growth"],
            "distractors": ["gravity", "magnetism", "friction", "inertia", "momentum", 
                          "digestion", "circulation", "mutation", "adaptation", "evolution"]
        }
        
        examples = []
        for i in range(num_samples):
            template = random.choice(templates)
            
            # Fill in template
            phenomenon = random.choice(science_vocab["phenomenon"])
            cause = random.choice(science_vocab["cause"])
            entity = random.choice(science_vocab["entity"])
            function = random.choice(science_vocab["function"])
            process = random.choice(science_vocab["process"])
            outcome = random.choice(science_vocab["outcome"])
            distractors = random.sample(science_vocab["distractors"], 3)
            
            question = template["question"].format(
                phenomenon=phenomenon, entity=entity, outcome=outcome
            )
            correct = template["correct"].format(
                cause=cause, function=function, process=process
            )
            choices = [d for d in distractors] + [correct]
            random.shuffle(choices)
            correct_idx = choices.index(correct)
            
            support = template["support"].format(
                cause=cause, phenomenon=phenomenon,
                entity=entity, function=function,
                process=process, outcome=outcome
            )
            
            examples.append(ReasoningExample(
                question=question,
                choices=choices,
                correct_answer=correct_idx,
                support_text=support,
                question_id=f"syn_{i}"
            ))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Combine question with choices for encoding
        text = f"{example.question} [SEP] " + " [SEP] ".join(example.choices)
        if example.support_text:
            text = f"{example.support_text} [SEP] {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example.correct_answer),
            "question_id": example.question_id
        }
    
    def get_raw_example(self, idx: int) -> ReasoningExample:
        """Get raw example without tokenization."""
        return self.examples[idx]


class ARCDataset(Dataset):
    """
    AI2 Reasoning Challenge dataset.
    
    Two difficulty levels:
    - Easy: Simpler questions
    - Challenge: Requires more complex reasoning
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        difficulty: str = "easy"  # "easy" or "challenge"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.difficulty = difficulty
        self.examples: List[ReasoningExample] = []
        
        self._load_data(max_samples)
    
    def _load_data(self, max_samples: Optional[int] = None):
        """Load and parse ARC data."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    
                    item = json.loads(line)
                    
                    # ARC format has choices with labels (A, B, C, D)
                    question = item.get("question", {})
                    choices_data = question.get("choices", [])
                    
                    choices = [c.get("text", "") for c in choices_data]
                    labels = [c.get("label", "") for c in choices_data]
                    
                    answer_key = item.get("answerKey", "A")
                    try:
                        correct_idx = labels.index(answer_key)
                    except ValueError:
                        correct_idx = 0
                    
                    example = ReasoningExample(
                        question=question.get("stem", ""),
                        choices=choices,
                        correct_answer=correct_idx,
                        question_id=item.get("id", str(i)),
                        difficulty=self.difficulty
                    )
                    self.examples.append(example)
        else:
            # Generate synthetic data
            self.examples = self._generate_arc_style_data(max_samples or 100)
    
    def _generate_arc_style_data(self, num_samples: int) -> List[ReasoningExample]:
        """Generate ARC-style reasoning questions."""
        # Multi-hop reasoning templates
        templates = [
            {
                "question": "If {premise1} and {premise2}, what can we conclude about {topic}?",
                "correct": "{conclusion}",
                "distractors": ["{wrong1}", "{wrong2}", "{wrong3}"],
                "reasoning": "multi_hop"
            },
            {
                "question": "A scientist observes that {observation}. Based on this, which hypothesis is most likely?",
                "correct": "{hypothesis}",
                "distractors": ["{wrong1}", "{wrong2}", "{wrong3}"],
                "reasoning": "abduction"
            },
            {
                "question": "What would happen if {condition}?",
                "correct": "{effect}",
                "distractors": ["{wrong1}", "{wrong2}", "{wrong3}"],
                "reasoning": "counterfactual"
            }
        ]
        
        vocab = {
            "premise1": ["temperature increases", "pressure decreases", "energy is added"],
            "premise2": ["volume is constant", "mass remains fixed", "no heat escapes"],
            "topic": ["the gas", "the reaction", "the system"],
            "conclusion": ["pressure increases", "rate increases", "entropy increases"],
            "observation": ["plants grow toward light", "metals conduct electricity", "ice floats on water"],
            "hypothesis": ["phototropism occurs", "electrons move freely", "ice is less dense"],
            "condition": ["gravity doubled", "friction eliminated", "photosynthesis stopped"],
            "effect": ["objects fall faster", "perpetual motion possible", "oxygen depleted"],
            "wrong": ["nothing happens", "opposite occurs", "no change", "random outcome",
                     "temperature drops", "mass increases", "time reverses"]
        }
        
        examples = []
        for i in range(num_samples):
            template = random.choice(templates)
            
            p1 = random.choice(vocab["premise1"])
            p2 = random.choice(vocab["premise2"])
            topic = random.choice(vocab["topic"])
            conclusion = random.choice(vocab["conclusion"])
            obs = random.choice(vocab["observation"])
            hyp = random.choice(vocab["hypothesis"])
            cond = random.choice(vocab["condition"])
            effect = random.choice(vocab["effect"])
            wrongs = random.sample(vocab["wrong"], 3)
            
            question = template["question"].format(
                premise1=p1, premise2=p2, topic=topic,
                observation=obs, condition=cond
            )
            correct = template["correct"].format(
                conclusion=conclusion, hypothesis=hyp, effect=effect
            )
            
            choices = wrongs + [correct]
            random.shuffle(choices)
            correct_idx = choices.index(correct)
            
            difficulty = "challenge" if template["reasoning"] in ["multi_hop", "counterfactual"] else "easy"
            
            examples.append(ReasoningExample(
                question=question,
                choices=choices,
                correct_answer=correct_idx,
                question_id=f"arc_{i}",
                difficulty=difficulty
            ))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format: question [SEP] A. choice1 [SEP] B. choice2 ...
        choice_text = " [SEP] ".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(example.choices)
        )
        text = f"{example.question} [SEP] {choice_text}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example.correct_answer),
            "question_id": example.question_id
        }
    
    def get_raw_example(self, idx: int) -> ReasoningExample:
        return self.examples[idx]


def get_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Get dataloader for specified dataset.
    
    Args:
        dataset_name: One of "sciq", "arc_easy", "arc_challenge", "synthetic"
        data_path: Path to data file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_samples: Optional limit on samples
        num_workers: DataLoader workers
        **kwargs: Additional dataset arguments
        
    Returns:
        DataLoader instance
    """
    tokenizer = get_tokenizer()
    
    if dataset_name == "sciq":
        dataset = SciQDataset(data_path, tokenizer, max_samples=max_samples, **kwargs)
    elif dataset_name in ["arc_easy", "arc_challenge"]:
        difficulty = "easy" if dataset_name == "arc_easy" else "challenge"
        dataset = ARCDataset(data_path, tokenizer, max_samples=max_samples, 
                            difficulty=difficulty, **kwargs)
    else:
        # Default to synthetic SciQ-style data
        dataset = SciQDataset(data_path, tokenizer, max_samples=max_samples, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


if __name__ == "__main__":
    # Test data loading
    print("Testing SciQ dataset...")
    dataset = SciQDataset("data/train.json", max_samples=5)
    print(f"Loaded {len(dataset)} examples")
    
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample label: {sample['labels']}")
    
    raw = dataset.get_raw_example(0)
    print(f"Question: {raw.question}")
    print(f"Choices: {raw.choices}")
