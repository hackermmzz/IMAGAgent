# IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection
[![Paper](https://img.shields.io/badge/Paper-IJCAI2026-blue)](https://arxiv.org/abs/xxx.xxxx)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-blueviolet)](environment.yml)

## Project Overview
IMAGAgent is a multi-turn image editing agent framework built on a **closed-loop "plan-execute-reflect" mechanism**. It addresses three core challenges in existing multi-turn image editing: **error accumulation**, **semantic drift**, and **structural distortion**. By deeply synergizing instruction parsing, tool scheduling, and adaptive correction, IMAGAgent achieves superior performance in instruction consistency, editing precision, and overall quality for long-horizon editing tasks.

### Core Advantages
- 🚀 **Constraint-Aware Planning**: Decomposes complex natural language instructions into atomic subtasks adhering to three key constraints—target singularity, semantic atomicity, and visual perceptibility—using vision-language models (VLMs).
- 🔧 **Dynamic Tool-Chain Orchestration**: Dynamically constructs execution paths based on the current image state, subtask requirements, and historical context to adaptively schedule heterogeneous vision tools (retrieval, segmentation, detection, editing).
- 🔄 **Multi-Expert Collaborative Reflection**: Integrates feedback from multiple VLM experts to generate fine-grained critiques, triggering self-correction loops and optimizing future decisions to suppress error propagation.
- 📊 **Dedicated Benchmark**: Introduces MTEditBench, a comprehensive dataset with 1,000 high-quality sequences (≥4 turns) designed for evaluating long-horizon multi-turn image editing stability.
### Framework
![framework](https://github.com/hackermmzz/IMAGAgent/blob/master/case/framework.jpg)

## Demo Results
### On MagicBrush
![Results on MagicBrush](https://github.com/hackermmzz/IMAGAgent/blob/master/case/MagicBrushCompare.jpg)

### On MTEditBench
![Results on MTEditBench](https://github.com/hackermmzz/IMAGAgent/blob/master/case/MTEditBenchCompare.jpg)

### Feedback
![Results on Feedback](https://github.com/hackermmzz/IMAGAgent/blob/master/case/feedback.jpg)

### Key Capability Demos
- Complex attribute editing: Convert airplane material to brick while preserving silhouette and replacing background with a forest clearing.
- Long-sequence stability: Retain object identity, geometry, and texture consistency after 5+ editing turns.
- Error correction: Automatically identify and fix unintended artifacts (e.g., spurious objects, structural deformation) in tasks like "remove power sockets".

## Quick Start
### Environment Setup (Conda)
This project uses a Conda environment defined in `environment.yml` for reproducible dependencies.

```bash
# Clone the repository
git clone https://github.com/hackermmzz/IMAGAgent.git
cd IMAGAgent

# Create and activate Conda environment from environment.yml
conda env create -f environment.yml
conda activate Edit

# Verify environment installation
conda list  # Check if all dependencies are installed correctly
```

### Environment Details
The `environment.yml` file includes all required dependencies (derived from Conda export), including:
- Core libraries: Python 3.9.13, PyTorch 2.6.0, TorchVision, TorchAudio
- Vision models/tools: Transformers, Diffusers, Segment-Anything (SAM), GroundingDINO
- VLMs/LLMs: Qwen-VL-MAX, GLM-4.1V-9B-Thinking, SAM3, Grounding-DINO-Base, CLIP-ViT-Large-Patch14,Doubao-Seedream-4.0, Qwen-Image-Edit, Stable-Diffusion-XL-Base-1.0, Doubao-Seed-1.6-Vision, Doubao-Seed-1.6, ,DeepSeek-V3.2

### Hardware Requirements
- GPU: NVIDIA A100 (recommended, for efficient inference)
- CPU: 16-core+ Intel/AMD processor
- RAM: 32GB+ (for model hosting and context management)
- CUDA: 12.4+ (compatible with PyTorch installation in environment.yml)

### Basic Usage
```bash
python run.py --img_path test.png --prompt "Have the girl in the picture strike a 'yes' pose." --dir output/
```

## Framework Architecture
IMAGAgent’s closed-loop pipeline consists of three core modules:

### 1. Constraint-Aware Planning Module
- Uses VLMs (Qwen-VL-Max) to ground instructions in the initial image’s spatial layout.
- Decomposes complex instructions into executable atomic subtasks following three constraints:
  - Target Singularity: Edit one entity/group at a time.
  - Semantic Atomicity: Subtasks cannot be further split without losing meaning.
  - Visual Perceptibility: Subtasks must produce tangible visual changes.
- Reorders subtasks based on semantic dependencies to ensure causal consistency.

### 2. Tool-Chain Orchestration Module
- Dynamically selects and schedules heterogeneous tools (e.g., SAM for segmentation, GroundingDINO for detection, Qwen Diffusion for editing) using GLM-4.1V-9B-Thinking.
- Leverages chain-of-thought (CoT) reasoning and historical context to construct optimal execution paths for each subtask.

### 3. Multi-Expert Collaborative Reflection Module
- Employs three VLM experts (Qwen, Doubao, etc.) to evaluate intermediate results across four dimensions: semantic alignment, perceptual quality, aesthetics, and logical consistency.
- Uses a central LLM (DeepSeek-V3.2) to aggregate expert feedback into a unified report (positive traits, negative defects, quantitative score).
- Triggers retries for low-score results or selects the best candidate after max iterations to ensure quality.

## Experiments
### Datasets
- **MTEditBench**: 1,000 multi-turn sequences (4-8+ turns) for long-horizon stability evaluation.
- **MagicBrush**: 10k+ instruction-image-edit triplets (max 3 turns) for baseline comparison.

### Evaluation Metrics
- **DINO**: Measures visual consistency and structural preservation.
- **CLIP-I**: Evaluates image semantic similarity to maintain identity.
- **CLIP-T**: Assesses cross-modal alignment between text instructions and edited images.

### Key Results
| Dataset | Metric | SOTA Baselines (OmniGen) | IMAGAgent |
|---------|--------|---------------------------------|-----------|
| MTEditBench (Avg 5 turns) | DINO | 0.671 | 0.766 |
| MTEditBench (Avg 5 turns) | CLIP-I | 0.825 | 0.875 |
| MagicBrush (Avg 3 turns) | CLIP-T | 0.266 | 0.282 |

- IMAGAgent outperforms all baselines, with performance advantages growing as the number of editing turns increases.
- Ablation studies confirm that closed-loop reflection, constraint-aware planning, historical context, and multi-expert collaboration are critical to performance.

## Citation
If you use IMAGAgent or MTEditBench in your research, please cite our paper:
```bibtex
@inproceedings{imagagent2026,
  title={IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection},
  author={Author Name},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## License
This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please reach out to [2049983474@qq.com].