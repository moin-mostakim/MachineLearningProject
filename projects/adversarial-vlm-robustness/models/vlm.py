"""Vision-Language Model combining vision and text encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .vision_encoder import VisionEncoder, get_vision_encoder
from .text_encoder import TextEncoder, get_text_encoder


class VisionLanguageModel(nn.Module):
    """Base Vision-Language Model with contrastive learning."""
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        text_encoder: TextEncoder,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Ensure output dimensions match
        assert vision_encoder.output_dim == text_encoder.output_dim, \
            f"Dimension mismatch: vision={vision_encoder.output_dim}, text={text_encoder.output_dim}"
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature space."""
        features = self.vision_encoder(images)
        return F.normalize(features, dim=-1)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature space."""
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1)
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing image and text embeddings."""
        image_features = self.encode_image(images)
        
        if text is not None:
            text_features = self.encode_text(text)
        else:
            # For classification, use all class embeddings
            text_features = self.text_encoder.encoder.get_all_embeddings()
            text_features = F.normalize(text_features, dim=-1)
        
        # Compute logits
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = logits_per_image.T
        
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "temperature": self.temperature
        }
    
    def get_classification_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Get classification logits for images."""
        output = self.forward(images)
        return output["logits_per_image"]


class SimpleCLIP(nn.Module):
    """Simplified CLIP-style model for CIFAR-10 classification.
    
    Uses contrastive learning between image and class label embeddings.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.vision_encoder = get_vision_encoder(config)
        self.text_encoder = get_text_encoder(config)
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # ln(100)
        
        self.num_classes = len(config.data.cifar10_labels)
        self.class_names = config.data.cifar10_labels
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized feature space."""
        features = self.vision_encoder(images)
        return F.normalize(features, dim=-1)
    
    def get_text_features(self) -> torch.Tensor:
        """Get normalized text features for all classes."""
        features = self.text_encoder.encoder.get_all_embeddings()
        return F.normalize(features, dim=-1)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits."""
        image_features = self.encode_image(images)
        text_features = self.get_text_features()
        
        # Compute similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        
        return logits
    
    def contrastive_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute contrastive loss for image-label pairs."""
        image_features = self.encode_image(images)
        text_features = self.get_text_features()
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        
        return loss, {"accuracy": acc.item(), "logit_scale": logit_scale.item()}


class SmoothCLIP(SimpleCLIP):
    """CLIP model with noise injection for certified training."""
    
    def __init__(self, config, sigma: float = 0.25):
        super().__init__(config)
        self.sigma = sigma
    
    def forward(self, images: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Forward with optional Gaussian noise for randomized smoothing."""
        if add_noise and self.training:
            noise = torch.randn_like(images) * self.sigma
            images = images + noise
            images = torch.clamp(images, 0, 1)
        
        return super().forward(images)
    
    def predict_with_smoothing(
        self,
        images: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """Predict using Monte Carlo sampling for randomized smoothing."""
        self.eval()
        batch_size = images.shape[0]
        
        # Accumulate votes
        votes = torch.zeros(batch_size, self.num_classes, device=images.device)
        
        with torch.no_grad():
            for _ in range(n_samples):
                noise = torch.randn_like(images) * self.sigma
                noisy_images = torch.clamp(images + noise, 0, 1)
                logits = super().forward(noisy_images)
                preds = logits.argmax(dim=-1)
                
                for i in range(batch_size):
                    votes[i, preds[i]] += 1
        
        return votes.argmax(dim=-1), votes / n_samples


def create_vlm(config) -> SimpleCLIP:
    """Factory function to create VLM from config."""
    if config.training.mode == "certified":
        return SmoothCLIP(config, sigma=config.certified.sigma)
    return SimpleCLIP(config)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }
    
    if hasattr(model, 'vision_encoder'):
        result["vision_encoder"] = sum(p.numel() for p in model.vision_encoder.parameters())
    if hasattr(model, 'text_encoder'):
        result["text_encoder"] = sum(p.numel() for p in model.text_encoder.parameters())
    
    return result


def get_parameter_table(model: nn.Module) -> str:
    """Generate parameter count table."""
    params = count_parameters(model)
    
    table = "\n" + "=" * 50 + "\n"
    table += "Model Parameter Count\n"
    table += "=" * 50 + "\n"
    table += f"{'Component':<25} {'Parameters':>15}\n"
    table += "-" * 50 + "\n"
    
    for key, value in params.items():
        if key != "non_trainable":
            table += f"{key:<25} {value:>15,}\n"
    
    table += "-" * 50 + "\n"
    table += f"{'Memory (float32)':<25} {params['total'] * 4 / 1e6:>12.2f} MB\n"
    table += f"{'Memory (float16)':<25} {params['total'] * 2 / 1e6:>12.2f} MB\n"
    table += "=" * 50 + "\n"
    
    return table


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/projects/adversarial-vlm-robustness")
    from config import get_config
    
    config = get_config(debug=True)
    model = create_vlm(config)
    
    print(get_parameter_table(model))
    
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
