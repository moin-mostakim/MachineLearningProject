"""
Data loading utilities for scientific reasoning datasets.

Datasets:
---------
SciQ: Science Question Answering
- Source: HuggingFace Hub (allenai/sciq)
- URL: https://huggingface.co/datasets/allenai/sciq
- Official: https://allenai.org/data/sciq
- License: CC BY-NC 3.0
- Size: ~13MB (13,679 questions: 11,679 train, 1,000 val, 1,000 test)
- Download: Automatic via datasets.load_dataset()

ARC: AI2 Reasoning Challenge
- Source: HuggingFace Hub (allenai/ai2_arc)
- URL: https://huggingface.co/datasets/allenai/ai2_arc
- Official: https://allenai.org/data/arc
- License: CC BY-SA 4.0
- Size: ~1.1MB (7,787 questions)

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

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Dataset configuration
DATASET_CONFIG = {
    "sciq": {
        "name": "SciQ",
        "huggingface": "allenai/sciq",
        "url": "https://huggingface.co/datasets/allenai/sciq",
        "official": "https://allenai.org/data/sciq",
        "license": "CC BY-NC 3.0",
        "size": "~13MB",
        "train_samples": 11679,
        "val_samples": 1000,
        "test_samples": 1000,
    },
    "arc": {
        "name": "AI2 Reasoning Challenge (ARC)",
        "huggingface": "allenai/ai2_arc",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "official": "https://allenai.org/data/arc",
        "license": "CC BY-SA 4.0",
        "size": "~1.1MB",
        "easy_samples": 5197,
        "challenge_samples": 2590,
    }
}


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


class HuggingFaceSciQDataset(Dataset):
    """
    SciQ dataset loaded directly from HuggingFace Hub.
    
    Source: https://huggingface.co/datasets/allenai/sciq
    License: CC BY-NC 3.0
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None
    ):
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library required for HuggingFace SciQ. "
                "Install with: pip install datasets"
            )
        
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.examples: List[ReasoningExample] = []
        
        print(f"Loading SciQ from HuggingFace Hub (split={split})...")
        dataset = load_dataset("allenai/sciq", split=split, )
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        for i, item in enumerate(dataset):
            choices = [
                item.get("distractor1", ""),
                item.get("distractor2", ""),
                item.get("distractor3", ""),
                item.get("correct_answer", "")
            ]
            random.shuffle(choices)
            correct_idx = choices.index(item.get("correct_answer", ""))
            
            example = ReasoningExample(
                question=item.get("question", ""),
                choices=choices,
                correct_answer=correct_idx,
                support_text=item.get("support", ""),
                question_id=f"sciq_{i}"
            )
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} SciQ examples from HuggingFace")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        text = f"{example.question} [SEP] " + " [SEP] ".join(example.choices)
        if example.support_text:
            text = f"{example.support_text} [SEP] {text}"
        
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


class HuggingFaceARCDataset(Dataset):
    """
    AI2 Reasoning Challenge (ARC) dataset loaded from HuggingFace Hub.
    
    Source: https://huggingface.co/datasets/allenai/ai2_arc
    License: CC BY-SA 4.0
    """
    
    def __init__(
        self,
        split: str = "train",
        difficulty: str = "easy",
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None
    ):
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library required for HuggingFace ARC. "
                "Install with: pip install datasets"
            )
        
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.difficulty = difficulty
        self.examples: List[ReasoningExample] = []
        
        config_name = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
        print(f"Loading ARC ({config_name}) from HuggingFace Hub (split={split})...")
        dataset = load_dataset("allenai/ai2_arc", config_name, split=split, )
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        for i, item in enumerate(dataset):
            choices = item.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            
            answer_key = item.get("answerKey", "A")
            try:
                correct_idx = choice_labels.index(answer_key)
            except ValueError:
                correct_idx = 0
            
            example = ReasoningExample(
                question=item.get("question", ""),
                choices=choice_texts,
                correct_answer=correct_idx,
                question_id=item.get("id", f"arc_{i}"),
                difficulty=difficulty
            )
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} ARC examples from HuggingFace")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
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


class SciQDataset(Dataset):
    """
    SciQ dataset for science question answering.
    
    Format: Multiple choice questions with 4 options and support text.
    
    By default loads real data from HuggingFace. Set use_synthetic=True
    for synthetic data generation.
    """
    
    def __init__(
        self,
        data_path: str = None,
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        use_synthetic: bool = False,
        split: str = "train"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.examples: List[ReasoningExample] = []
        
        if use_synthetic:
            print("Using synthetic SciQ data (opt-in)")
            self.examples = self._generate_synthetic_data(max_samples or 100)
        elif HAS_DATASETS:
            self._load_from_huggingface(split, max_samples)
        elif data_path and os.path.exists(data_path):
            self._load_from_file(max_samples)
        else:
            raise RuntimeError(
                "Cannot load SciQ data. Options:\n"
                "1. Install datasets library: pip install datasets\n"
                "2. Provide a valid data_path to local JSON file\n"
                "3. Use use_synthetic=True for synthetic data"
            )
    
    def _load_from_huggingface(self, split: str, max_samples: Optional[int]):
        """Load SciQ from HuggingFace Hub."""
        print(f"Loading SciQ from HuggingFace Hub (split={split})...")
        dataset = load_dataset("allenai/sciq", split=split, )
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        for i, item in enumerate(dataset):
            choices = [
                item.get("distractor1", ""),
                item.get("distractor2", ""),
                item.get("distractor3", ""),
                item.get("correct_answer", "")
            ]
            random.shuffle(choices)
            correct_idx = choices.index(item.get("correct_answer", ""))
            
            example = ReasoningExample(
                question=item.get("question", ""),
                choices=choices,
                correct_answer=correct_idx,
                support_text=item.get("support", ""),
                question_id=f"sciq_{i}"
            )
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} SciQ examples from HuggingFace")
    
    def _load_from_file(self, max_samples: Optional[int] = None):
        """Load and parse SciQ data from local file."""
        print(f"Loading SciQ from local file: {self.data_path}")
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
                correct_answer=3,
                support_text=item.get("support", ""),
                question_id=str(i)
            )
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from local file")
    
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
    
    By default loads real data from HuggingFace. Set use_synthetic=True
    for synthetic data generation.
    """
    
    def __init__(
        self,
        data_path: str = None,
        tokenizer=None,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        difficulty: str = "easy",
        use_synthetic: bool = False,
        split: str = "train"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        self.difficulty = difficulty
        self.examples: List[ReasoningExample] = []
        
        if use_synthetic:
            print(f"Using synthetic ARC data (opt-in)")
            self.examples = self._generate_arc_style_data(max_samples or 100)
        elif HAS_DATASETS:
            self._load_from_huggingface(split, max_samples)
        elif data_path and os.path.exists(data_path):
            self._load_from_file(max_samples)
        else:
            raise RuntimeError(
                "Cannot load ARC data. Options:\n"
                "1. Install datasets library: pip install datasets\n"
                "2. Provide a valid data_path to local JSONL file\n"
                "3. Use use_synthetic=True for synthetic data"
            )
    
    def _load_from_huggingface(self, split: str, max_samples: Optional[int]):
        """Load ARC from HuggingFace Hub."""
        config_name = "ARC-Easy" if self.difficulty == "easy" else "ARC-Challenge"
        print(f"Loading ARC ({config_name}) from HuggingFace Hub (split={split})...")
        dataset = load_dataset("allenai/ai2_arc", config_name, split=split, )
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        for i, item in enumerate(dataset):
            choices = item.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            
            answer_key = item.get("answerKey", "A")
            try:
                correct_idx = choice_labels.index(answer_key)
            except ValueError:
                correct_idx = 0
            
            example = ReasoningExample(
                question=item.get("question", ""),
                choices=choice_texts,
                correct_answer=correct_idx,
                question_id=item.get("id", f"arc_{i}"),
                difficulty=self.difficulty
            )
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} ARC examples from HuggingFace")
    
    def _load_from_file(self, max_samples: Optional[int] = None):
        """Load and parse ARC data from local file."""
        print(f"Loading ARC from local file: {self.data_path}")
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                item = json.loads(line)
                
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
        
        print(f"Loaded {len(self.examples)} examples from local file")
    
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
    batch_size: int,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    use_synthetic: bool = False,
    split: str = "train",
    data_path: str = None,
    **kwargs
) -> DataLoader:
    """
    Get dataloader for specified dataset.
    
    Args:
        dataset_name: One of "sciq", "arc_easy", "arc_challenge"
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_samples: Optional limit on samples
        num_workers: DataLoader workers
        use_synthetic: If True, use synthetic data instead of real (default False)
        split: Dataset split ("train", "validation", "test")
        data_path: Optional path to local data file
        **kwargs: Additional dataset arguments
        
    Returns:
        DataLoader instance
    """
    tokenizer = get_tokenizer()
    
    if dataset_name == "sciq":
        dataset = SciQDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            split=split,
            **kwargs
        )
    elif dataset_name in ["arc_easy", "arc_challenge"]:
        difficulty = "easy" if dataset_name == "arc_easy" else "challenge"
        dataset = ARCDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_samples=max_samples,
            difficulty=difficulty,
            use_synthetic=use_synthetic,
            split=split,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'sciq', 'arc_easy', or 'arc_challenge'")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


if __name__ == "__main__":
    # Test data loading with real HuggingFace data
    print("=" * 60)
    print("Testing SciQ dataset from HuggingFace...")
    print("=" * 60)
    
    try:
        dataset = SciQDataset(max_samples=5)
        print(f"Loaded {len(dataset)} examples")
        
        sample = dataset[0]
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample label: {sample['labels']}")
        
        raw = dataset.get_raw_example(0)
        print(f"Question: {raw.question}")
        print(f"Choices: {raw.choices}")
        print(f"Correct answer index: {raw.correct_answer}")
    except Exception as e:
        print(f"Error loading real data: {e}")
    
    print("\n" + "=" * 60)
    print("Testing ARC dataset from HuggingFace...")
    print("=" * 60)
    
    try:
        arc_dataset = ARCDataset(difficulty="easy", max_samples=5)
        print(f"Loaded {len(arc_dataset)} ARC examples")
        
        raw_arc = arc_dataset.get_raw_example(0)
        print(f"Question: {raw_arc.question}")
        print(f"Choices: {raw_arc.choices}")
    except Exception as e:
        print(f"Error loading ARC data: {e}")
    
    print("\n" + "=" * 60)
    print("Testing synthetic data (opt-in)...")
    print("=" * 60)
    
    synthetic = SciQDataset(use_synthetic=True, max_samples=3)
    print(f"Generated {len(synthetic)} synthetic examples")
    raw_syn = synthetic.get_raw_example(0)
    print(f"Synthetic question: {raw_syn.question}")
