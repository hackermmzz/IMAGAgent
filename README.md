# IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection
[![Paper](https://img.shields.io/badge/Paper-IJCAI2025-blue)](https://arxiv.org/abs/xxx.xxxx)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Project Overview
IMAGAgent is a multi-turn image editing agent framework built on a **closed-loop "plan-execute-reflect" mechanism**. It addresses three core challenges in existing multi-turn image editing: **error accumulation**, **semantic drift**, and **structural distortion**. By deeply synergizing instruction parsing, tool scheduling, and adaptive correction, IMAGAgent achieves superior performance in instruction consistency, editing precision, and overall quality for long-horizon editing tasks.

### Core Advantages
- 🚀 **Constraint-Aware Planning**: Decomposes complex natural language instructions into atomic subtasks adhering to three key constraints—target singularity, semantic atomicity, and visual perceptibility—using vision-language models (VLMs).
- 🔧 **Dynamic Tool-Chain Orchestration**: Dynamically constructs execution paths based on the current image state, subtask requirements, and historical context to adaptively schedule heterogeneous vision tools (retrieval, segmentation, detection, editing).
- 🔄 **Multi-Expert Collaborative Reflection**: Integrates feedback from multiple VLM experts to generate fine-grained critiques, triggering self-correction loops and optimizing future decisions to suppress error propagation.
- 📊 **Dedicated Benchmark**: Introduces MTEditBench, a comprehensive dataset with 1,000 high-quality sequences (≥4 turns) designed for evaluating long-horizon multi-turn image editing stability.

## Demo Results
### Multi-Turn Editing Comparison (MTEditBench)
| Editing Turns | Instruction Description | Baselines (GPT-4o/VINCIE/OmniGen) | IMAGAgent |
|---------------|-------------------------|-----------------------------------|-----------|
| 1-5           | Remove person → Add cat on chair → Pink chair → Urban park background → Comic style | Suffer from semantic drift and structural distortion | Maintain consistent semantics and visual integrity throughout |

### Key Capability Demos
- Complex attribute editing: Convert airplane material to brick while preserving silhouette and replacing background with a forest clearing.
- Long-sequence stability: Retain object identity, geometry, and texture consistency after 5+ editing turns.
- Error correction: Automatically identify and fix unintended artifacts (e.g., spurious objects, structural deformation) in tasks like "remove power sockets".

## Quick Start
### Environment Setup
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow opencv-python scikit-image clip diffusers
pip install qwen-vl glm4-vl deepseek-llm  # VLMs/LLMs for planning/orchestration/reflection
pip install groundingdino-py segment-anything  # Detection/segmentation tools
```

### Hardware Requirements
- GPU: NVIDIA A100 (recommended, for efficient inference)
- Minimum: NVIDIA RTX 3090/4090 (16GB+ VRAM)
- CPU: 16-core+ Intel/AMD processor
- RAM: 32GB+ (for model hosting and context management)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/IMAGAgent.git
cd IMAGAgent

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models and datasets (see Data & Models section)
bash scripts/download_resources.sh
```

### Basic Usage
```python
from imagagent import IMAGAgent

# Initialize the agent
agent = IMAGAgent(
    success_threshold=7.0,  # τ_sr: minimum score to pass a subtask
    max_iterations=3        # τ_it: maximum retries per subtask
)

# Load initial image and multi-turn instructions
initial_image = "input.jpg"
instructions = [
    "Remove the person from the chair",
    "Add a cat sitting on the chair",
    "Change the chair color to vibrant pink",
    "Replace the background with an urban park scene",
    "Change the image style to comic book art"
]

# Run multi-turn editing
final_image = agent.edit(initial_image, instructions)
final_image.save("output.jpg")
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
| Dataset | Metric | SOTA Baselines (GPT-4o/OmniGen) | IMAGAgent |
|---------|--------|---------------------------------|-----------|
| MTEditBench (Avg 5 turns) | DINO | 0.727 | 0.766 |
| MTEditBench (Avg 5 turns) | CLIP-I | 0.838 | 0.875 |
| MagicBrush (Avg 3 turns) | CLIP-T | 0.275 | 0.282 |

- IMAGAgent outperforms all baselines, with performance advantages growing as the number of editing turns increases.
- Ablation studies confirm that closed-loop reflection, constraint-aware planning, historical context, and multi-expert collaboration are critical to performance.

## Data & Models
### Download Resources
```bash
# Download MTEditBench dataset (10GB)
bash scripts/download_mteditbench.sh

# Download MagicBrush dataset (8GB)
bash scripts/download_magicbrush.sh

# Download pre-trained VLMs/LLMs (requires Hugging Face access)
bash scripts/download_models.sh
```

### Dataset Structure
```
data/
├── MTEditBench/
│   ├── sequences/  # 1000 folders of multi-turn editing sequences
│   ├── annotations/ # Instruction and metadata JSON files
│   └── splits/      # Train/val/test splits
└── MagicBrush/
    ├── images/      # Original and edited images
    └── annotations/ # Instruction triplets
```

## Citation
If you use IMAGAgent or MTEditBench in your research, please cite our paper:
```bibtex
@inproceedings{imagagent2025,
  title={IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection},
  author={Author Name},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```

## License
This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please reach out to [email@example.com].

---

Would you like me to refine any section (e.g., add more code examples, expand the experiment details, or optimize the installation instructions)? I can also help generate a `requirements.txt` file or additional bash scripts for resource downloading.