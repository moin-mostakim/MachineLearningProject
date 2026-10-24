# Machine Learning Research Projects

A collection of 15 machine learning research implementations covering deep learning, computer vision, NLP, medical imaging, and scientific computing. Each project is designed with modular architecture, CPU-first execution, INT8 quantization support, and comprehensive resource monitoring.

## Repository Overview

This repository contains implementations of cutting-edge ML research across diverse domains:
- **Medical AI**: Diabetic retinopathy detection, neurodegenerative disease classification, patient monitoring
- **Scientific ML**: Climate emulation, materials discovery, autonomous driving
- **Foundation Models**: LLM fine-tuning, vision transformers, multimodal learning
- **Efficient ML**: Neural architecture search, continual learning, federated learning

## Project Index

| # | Title | Branch | Research Problem | Dataset(s) | Model | Params |
|---|-------|--------|------------------|------------|-------|--------|
| 01 | [LoRA Domain LLM Fine-tuning](https://github.com/moin-mostakim/MachineLearningProject/tree/project-01-lora-domain-llm-finetuning) | `project-01-lora-domain-llm-finetuning` | Parameter-efficient LLM adaptation | [WikiText-2](https://huggingface.co/datasets/wikitext) | Adaptive LoRA + DistilGPT-2 | ~82M |
| 02 | [Multimodal Neurodegenerative Detection](https://github.com/moin-mostakim/MachineLearningProject/tree/project-02-multimodal-neurodegenerative-detection) | `project-02-multimodal-neurodegenerative-detection` | Early Alzheimer's detection from brain imaging + clinical data | [OASIS-3](https://www.oasis-brains.org/), [ADNI](http://adni.loni.usc.edu/) | Attention-based multimodal fusion | ~300K |
| 03 | [Causal Autonomous Driving](https://github.com/moin-mostakim/MachineLearningProject/tree/project-03-causal-autonomous-driving) | `project-03-causal-autonomous-driving` | Robust decision-making under distribution shift | [nuScenes](https://www.nuscenes.org/), [KITTI](http://www.cvlibs.net/datasets/kitti/) | IRM + CNN | ~200K |
| 04 | [Federated Learning Privacy](https://github.com/moin-mostakim/MachineLearningProject/tree/project-04-federated-learning-privacy) | `project-04-federated-learning-privacy` | Privacy-preserving distributed training | [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10), [FEMNIST](https://github.com/TalwalkarLab/leaf) | FedAvg/FedProx + DP-SGD | ~100K |
| 05 | [Generative Materials Design](https://github.com/moin-mostakim/MachineLearningProject/tree/project-05-generative-materials-design) | `project-05-generative-materials-design` | Inverse molecular design with multi-objective optimization | [QM9](https://pytorch-geometric.readthedocs.io/) | Molecular VAE + GNN | ~500K |
| 06 | [Neurosymbolic Reasoning](https://github.com/moin-mostakim/MachineLearningProject/tree/project-06-neurosymbolic-reasoning) | `project-06-neurosymbolic-reasoning` | Scientific hypothesis generation/verification | [SciQ](https://huggingface.co/datasets/allenai/sciq), [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | Neural+Symbolic+KG | ~11.2M |
| 07 | [Adversarial VLM Robustness](https://github.com/moin-mostakim/MachineLearningProject/tree/project-07-adversarial-vlm-robustness) | `project-07-adversarial-vlm-robustness` | Vision-language model robustness against attacks | [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10) | CLIP-style VLM | ~2M |
| 08 | [Continual Learning Sparse Expansion](https://github.com/moin-mostakim/MachineLearningProject/tree/project-08-continual-learning-sparse-expansion) | `project-08-continual-learning-sparse-expansion` | Lifelong learning without catastrophic forgetting | [CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100), [MNIST](https://huggingface.co/datasets/ylecun/mnist) | EWC + Dynamic Sparse | ~1.5M |
| 09 | [Edge AI Efficiency](https://github.com/moin-mostakim/MachineLearningProject/tree/project-09-edge-ai-efficiency) | `project-09-edge-ai-efficiency` | NAS + QAT for efficient edge deployment | [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10), [ImageNette](https://github.com/fastai/imagenette) | MobileNetV2 + NAS | ~2.3M |
| 10 | [Climate AI Emulation](https://github.com/moin-mostakim/MachineLearningProject/tree/project-10-climate-ai-emulation) | `project-10-climate-ai-emulation` | Neural surrogate for climate modeling with uncertainty | [WeatherBench2](https://github.com/google-research/weatherbench2), [ERA5](https://cds.climate.copernicus.eu/) | U-Net + ConvLSTM + Ensemble | ~5M |
| 11 | [CS Tutoring LLM](https://github.com/moin-mostakim/MachineLearningProject/tree/project-11-cs-tutoring-llm) | `project-11-cs-tutoring-llm` | Intelligent tutoring with RAG for code explanation | [CodeSearchNet](https://huggingface.co/datasets/code_search_net) | CodeT5 + BM25 + RAG | ~60M |
| 12 | [Diabetic Retinopathy ViT](https://github.com/moin-mostakim/MachineLearningProject/tree/project-12-diabetic-retinopathy-vit) | `project-12-diabetic-retinopathy-vit` | Multi-scale explainable DR grading | [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)* | Multi-scale Lesion-Aware ViT | ~12M |
| 13 | [KAN Diabetic Retinopathy](https://github.com/moin-mostakim/MachineLearningProject/tree/project-13-kan-diabetic-retinopathy) | `project-13-kan-diabetic-retinopathy` | Kolmogorov-Arnold Networks for DR diagnosis | [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)* | Hybrid CNN+KAN | ~11.6M |
| 14 | [Multitask DR Segmentation](https://github.com/moin-mostakim/MachineLearningProject/tree/project-14-multitask-dr-segmentation) | `project-14-multitask-dr-segmentation` | Joint lesion segmentation + DR grading | [IDRiD](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)* | Multi-task U-Net | ~8M |
| 15 | [IoT Remote Patient Monitoring](https://github.com/moin-mostakim/MachineLearningProject/tree/project-15-iot-remote-patient-monitoring) | `project-15-iot-remote-patient-monitoring` | Real-time anomaly detection for vital signs | [MIMIC-III Demo](https://physionet.org/content/mimiciii-demo/) | LSTM/Transformer Autoencoder | ~500K |

*\* Requires Kaggle API key*

## INT8 Quantization Results Summary

All projects include INT8 dynamic quantization comparison:

| Project | FP32 Size | INT8 Size | Compression | Accuracy Drop |
|---------|-----------|-----------|-------------|---------------|
| 01 - LoRA LLM | ~313 MB | ~100 MB | 3.1x | <1% |
| 06 - Neurosymbolic | ~46 MB | ~31 MB | 1.5x | 0% |
| 07 - VLM | ~8 MB | ~3 MB | 2.7x | <1% |
| 09 - Edge AI | ~9 MB | ~2.5 MB | 3.6x | <1% |
| 10 - Climate | ~20 MB | ~6 MB | 3.3x | <2% |

## Common Requirements

All projects share a consistent structure:

### Architecture
- **Modular code**: Separate config, models, data, training, evaluation
- **CPU-first**: Optimized for CPU execution (GPU optional)
- **INT8 quantization**: Dynamic quantization for deployment
- **Resource monitoring**: CPU%, memory, wall time tracking
- **Hyperparameter config**: All parameters documented with literature justifications

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
```

### Optional (project-specific)
```bash
pip install torch-geometric  # Projects 05, 06
pip install sentencepiece    # Projects 01, 11
pip install kaggle           # Projects 12, 13, 14
```

## How to Run a Project

```bash
# Clone the repository
git clone https://github.com/moin-mostakim/MachineLearningProject.git
cd MachineLearningProject

# Checkout a project branch
git checkout project-01-lora-domain-llm-finetuning

# Install dependencies
pip install -r requirements.txt

# Run in debug mode (quick test)
python main.py --debug

# Run full training
python main.py
```

## Dataset Requirements

### Open Access (No Credentials)
| Dataset | Projects | Download |
|---------|----------|----------|
| WikiText-2 | 01 | `load_dataset("wikitext")` |
| CIFAR-10/100 | 04, 07, 08, 09 | `torchvision.datasets.CIFAR10` |
| MNIST | 08 | `torchvision.datasets.MNIST` |
| SciQ, ARC | 06 | `load_dataset("allenai/sciq")` |
| CodeSearchNet | 11 | `load_dataset("code_search_net")` |
| QM9 | 05 | `torch_geometric.datasets.QM9` |
| MIMIC-III Demo | 15 | [PhysioNet](https://physionet.org/content/mimiciii-demo/) |
| OASIS | 02 | [oasis-brains.org](https://www.oasis-brains.org/) |

### Requires Kaggle API Key
| Dataset | Projects | Command |
|---------|----------|---------|
| APTOS 2019 | 12, 13 | `kaggle competitions download -c aptos2019-blindness-detection` |
| IDRiD | 14 | `kaggle datasets download -d tanlikesmath/diabetic-retinopathy-resized` |

Set Kaggle credentials:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Requires Registration/Agreement
| Dataset | Projects | Access |
|---------|----------|--------|
| ADNI | 02 | [adni.loni.usc.edu](http://adni.loni.usc.edu/) (DUA required) |
| nuScenes | 03 | [nuscenes.org](https://www.nuscenes.org/) (Free registration) |
| WeatherBench2/ERA5 | 10 | [WeatherBench2](https://github.com/google-research/weatherbench2), [CDS](https://cds.climate.copernicus.eu/) |
| MIMIC-III Full | 15 | [PhysioNet](https://physionet.org/content/mimiciii/) (CITI training) |

## Status Notes

### Ran on Real Data
- **06**: SciQ/ARC from HuggingFace ✓
- **07**: CIFAR-10 from HuggingFace ✓

### Documented for Real Data (Downloads on First Run)
- **01, 04, 08, 09, 11**: Auto-download from torchvision/HuggingFace
- **05**: QM9 auto-download via PyTorch Geometric

### Requires Manual Download
- **02**: OASIS-3 (free registration)
- **03**: nuScenes mini (free registration)
- **10**: WeatherBench2/ERA5 (large downloads)
- **12, 13, 14**: Kaggle API key required
- **15**: PhysioNet registration (demo is open)

### Uses Synthetic Data (Real Available)
All projects include synthetic data generation for pipeline testing. Synthetic data should NOT be used for meaningful experiments - real datasets are required for valid research.

## License

MIT License - See individual project directories for specific licenses.

## Citation

If you use code from this repository, please cite the relevant papers listed in each project's README.

## Contributing

Contributions are welcome! Please ensure any PRs maintain:
1. Modular project structure
2. CPU-first execution
3. INT8 quantization comparison
4. Resource monitoring
5. Literature-justified hyperparameters
