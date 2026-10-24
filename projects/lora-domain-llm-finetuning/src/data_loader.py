"""Data loading and preprocessing module."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainDataLoader:
    """Data loader for domain-specific text data."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_length: int = 128,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(
        self,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load and preprocess the dataset."""
        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        
        try:
            dataset = load_dataset(self.dataset_name, self.dataset_config)
        except Exception:
            dataset = load_dataset("Salesforce/wikitext", self.dataset_config)
        
        def tokenize_function(examples):
            texts = [t for t in examples["text"] if t.strip()]
            if not texts:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        processed = {}
        
        for split, max_samples in [
            ("train", max_train_samples),
            ("validation", max_val_samples),
            ("test", max_test_samples)
        ]:
            if split in dataset:
                split_data = dataset[split]
                if max_samples and len(split_data) > max_samples:
                    split_data = split_data.select(range(max_samples))
                
                processed[split] = split_data.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=split_data.column_names
                )
                processed[split] = processed[split].filter(
                    lambda x: len(x["input_ids"]) > 0
                )
                logger.info(f"Processed {split}: {len(processed[split])} samples")
        
        return processed
    
    def create_dataloaders(
        self,
        processed_data: Dict[str, Any],
        batch_size: int = 8
    ) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders."""
        
        def collate_fn(batch):
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            labels = torch.tensor([item["labels"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        dataloaders = {}
        for split, data in processed_data.items():
            shuffle = (split == "train")
            dataloaders[split] = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn
            )
        
        return dataloaders
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer


def load_domain_dataset(
    domain: str = "scientific",
    max_samples: int = 1000
) -> Dict[str, Any]:
    """Load domain-specific dataset based on domain name."""
    
    domain_datasets = {
        "scientific": ("scientific_papers", "arxiv"),
        "medical": ("pubmed_qa", "pqa_labeled"),
        "legal": ("pile-of-law/pile-of-law", "r_legaladvice"),
        "general": ("wikitext", "wikitext-2-raw-v1")
    }
    
    if domain in domain_datasets:
        ds_name, ds_config = domain_datasets[domain]
    else:
        ds_name, ds_config = "wikitext", "wikitext-2-raw-v1"
    
    try:
        dataset = load_dataset(ds_name, ds_config, split="train")
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        return {"name": ds_name, "config": ds_config, "data": dataset}
    except Exception as e:
        logger.warning(f"Failed to load {ds_name}: {e}. Falling back to wikitext.")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        return {"name": "wikitext", "config": "wikitext-2-raw-v1", "data": dataset}
