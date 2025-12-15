# Fine-Tuning Large Language Models for Medical Vietnamese–English Machine Translation

## 1. Data Preparation

### 1.1 Dataset Description
- Large-scale medical Vietnamese–English parallel corpus (~1M sentence pairs)
- Covers multiple subdomains (clinical notes, diagnoses, instructions)

### 1.2 Data Handling Strategy
- Due to dataset scale and hardware constraints, the data is split into
  multiple chunks of equal size
- Chunks are streamed sequentially during training

**Justification**
- Enables efficient training under limited GPU memory
- Allows stable optimization with QLoRA
- Facilitates curriculum-style exposure to domain diversity

---

## 2. Model Selection

### 2.1 Candidate Models
We consider compact instruction-tuned models suitable for constrained
medical MT:

- Qwen3-1.7B
- Qwen2.5-3B

### 2.2 Rationale for Model Choice
- Models are selected based on:
  - Parameter efficiency
  - Strong multilingual capability
  - Compatibility with parameter-efficient fine-tuning (QLoRA)

Larger models are excluded due to computational constraints, while smaller
models show limited capacity for medical terminology.

---

## 3. Evaluation Metrics

The following automatic metrics are used throughout the study:

- **BLEU** – lexical overlap (legacy reference)
- **chrF++** – character-level adequacy and morphology
- **COMET** – semantic adequacy and fluency (primary metric)
- **Perplexity** – model confidence and convergence behavior

---

## 4. Fine-Tuning Methodology

### 4.1 Fine-Tuning Strategy

- Parameter-efficient fine-tuning using **QLoRA**
- Base model parameters are frozen
- Only low-rank adapters are updated

**Justification**
- Reduces memory usage and training cost
- Preserves general language capability
- Enables scalable training on large datasets

---

### 4.2 Curriculum-Style Training

- Training data is processed sequentially in fixed-size chunks
- Evaluation is performed after each chunk for monitoring only

**Justification**
- Chunk-level metrics are noisy and not used for early stopping
- Improvements accumulate across chunks
- Prevents bias toward early data subsets

---

### 4.3 Parameter Settings

- Hyper-parameters are initially selected on a 100k subset
- Selection criteria:
  - Training stability
  - Convergence speed
  - Validation COMET score

Once selected, parameters are fixed for full-scale training to avoid
overfitting through excessive tuning.

---

## 5. Error Analysis

### 5.1 Motivation
Standard MT metrics do not explicitly capture clinically critical errors.
We therefore conduct targeted error analysis to identify high-risk failure
modes.

### 5.2 Error Categories
Errors are grouped into the following categories:

- Terminology errors
- Negation and polarity errors
- Dosage and numerical errors
- Hallucination and omission

### 5.3 Detection Method
- Errors are identified using automatic, weakly supervised heuristics
- A subset of detected samples is manually validated

---

## 6. Reinforcement Learning for Error Reduction

### 6.1 RL Objective
Reinforcement learning is applied to explicitly penalize clinically unsafe
errors while preserving translation quality.

### 6.2 Reward Design
The reward function combines:
- COMET score
- Penalties for detected error types

### 6.3 Training Setup
- RL is applied only to a subset of difficult samples
- Base model remains frozen
- Only LoRA parameters are updated

---

## 7. Evaluation and Comparisons

### 7.1 Automatic Metrics
Models are evaluated before and after RL using:
- BLEU
- chrF++
- COMET
- Perplexity

### 7.2 Error Rate Comparison
- Clinical error rates are compared pre- and post-RL
- Improvements are analyzed per error category

---

## 8. Summary
This work presents a scalable and safety-aware approach to medical machine
translation, combining parameter-efficient fine-tuning, curriculum-style
training, targeted error analysis, and reinforcement learning.
