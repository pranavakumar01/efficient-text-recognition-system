# 📘 Phase-I Technical Report & Defense Manual

**Institution:** N.M.A.M. Institute of Technology, Nitte – 574 110, Karnataka, India  
*(Department of Information Science & Engineering | ISO 9001:2015 Certified | Accredited with 'A' Grade by NAAC)*  
**Project Code:** 42  
**Project Title:** *Efficient Text Recognition from Images Using Deep Learning and Self Supervised Learning*  
**Guide:** Dr. Rashmi Naveen, Assistant Professor, Dept. of ISE  
**Team Members:**  
- Vishnumoorthy Raghavendra Bhat (`NNM23IS215`)  
- Tejaswi Shankar P (`NNM23IS194`)  
- Pranava Kumara K (`NNM23IS132`)  
- Simran Kasim Sayed (`NNM23IS179`)  

---

## 1. Executive Summary & Project Objectives

Text recognition from images is a fundamental problem in computer vision and pattern recognition, playing a vital role in applications such as document digitization, automatic form processing, assistive technologies, and scene text understanding. Traditional Optical Character Recognition (OCR) systems rely heavily on handcrafted features and rule-based techniques, which often fail when dealing with noisy images, complex backgrounds, or handwritten text.

Recent advances in deep learning have significantly improved OCR performance by enabling automatic feature extraction and sequence modeling. However, most deep learning–based approaches require large amounts of labeled training data and high computational resources, making them difficult to deploy in low-resource environments.

### Core Objectives:
1. **Model Implementation & Comparison:** Implement an OCR system using CNN and Transformer-based methods and compare their empirical performance.
2. **Robustness:** Improve recognition accuracy for handwritten, mathematical, and complex text.
3. **Computational Optimization:** Develop an optimized OCR system with lower computational complexity suitable for edge environments.
4. **Data Efficiency:** Reduce dependency on large labeled datasets using self-supervised learning (SSL) techniques.

---

## 2. Comprehensive Literature Review & Identified Research Gaps

A comprehensive survey of 30 landmark papers (including Feng et al. 2024, Yang et al. 2025, Li et al. 2021/2022, Kim et al. 2022 Donut, Xu et al. LayoutLMv2/v3, and He et al. MAE-OCR) identified **6 critical research gaps**:

1. **Lack of a Fully Integrated End-to-End Self-Supervised OCR Framework:** Existing research evaluates visual SSL, sequence modeling, and language post-processing in isolation rather than in a unified closed loop.
2. **Limited Use of SSL on Real-World Unlabeled Text Images:** Most SSL studies rely on synthetic data rather than degraded, curved, or mathematical text.
3. **Inefficient Learning with High Dependence on Labeled Data:** Few studies systematically quantify how much labeled data can be eliminated with contrastive pre-training.
4. **Limited Evaluation of Computational Efficiency & Deployment Feasibility:** Massive Vision Transformers (>62M params) prioritize accuracy but suffer from extreme latency and memory footprints.
5. **Weak Integration of Language Context in SSL OCR:** Visual representations are frequently uncoupled from lexical and mathematical syntax rules.
6. **Insufficient Benchmarking on Diverse Scenarios:** Lack of cross-domain evaluation across printed, handwritten, mathematical (LaTeX), and historical documents.

*(See [docs/literature_survey.md](file:///d:/Major%20Project/docs/literature_survey.md) for the full 30-paper review table and gap breakdown).*

---

## 3. System Architecture & Methodology

```text
[Input Document / Image]
       ↓
[Preprocessing Engine] (Grayscale, CLAHE Contrast, Bilateral Filter, Deskewing, Dynamic Aspect Ratio Scaling)
       ↓
[Line Segmentation Engine] (Projection Profiles, Horizontal Dilation, Overlapping Contour Clustering)
       ↓
┌───────────────────────────────────────┬──────────────────────────────────────┐
│ Primary Model (CNN + BiLSTM + Attn)  │ Baseline Comparison (TrOCR / ViT)   │
│  • ResNet Backbone Feature Extraction │  • Vision Transformer Encoder        │
│  • Bidirectional LSTM Sequence Model  │  • Autoregressive RoBERTa Decoder    │
│  • Bahdanau Attention Modulation      │  • Beam Search Generation            │
│  • CTC Prefix Beam Search Decoder     │                                      │
└───────────────────────────────────────┴──────────────────────────────────────┘
       ↓
[Linguistic & Math Post-Processing Engine] (OCR Confusion Matrix, Word Repair, LaTeX Protection)
       ↓
[Evaluation & Analytics Dashboard] (CER, WER, Latency ms, Parameter Count, Throughput FPS)
```

### Key Mathematical Formulations:

1. **Connectionist Temporal Classification (CTC) Loss**:
   $$\mathcal{L}_{CTC} = - \ln P(\mathbf{y} | \mathbf{x}) = - \ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} P(\pi | \mathbf{x})$$

2. **Bahdanau Temporal Attention Modulation**:
   $$e_{t,i} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i), \quad \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$
   $$\mathbf{h}'_t = \mathbf{h}_t \cdot (1.0 + \alpha_t)$$

3. **SimCLR InfoNCE Contrastive Loss (Self-Supervised Pretraining)**:
   $$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

---

## 4. Empirical Benchmark Results

### Empirical Comparison Summary Table:

| Metric | Primary Model (CNN + BiLSTM + Attention) | Baseline Model (Vision Transformer / TrOCR) |
| :--- | :---: | :---: |
| **Model Parameters** | **5,281,505 (~5.28M)** | 62,000,000 (~62M) |
| **Memory Efficiency** | **~21.1 MB (11.7x lighter)** | ~248.0 MB |
| **Average Latency (ms/img)** | **~65 - 120 ms (CPU)** | ~550 - 2,500 ms (CPU) |
| **Throughput (FPS)** | **10 - 15 FPS** | 0.4 - 1.8 FPS |
| **Mathematical / Complex CER** | **0.5348** | 1.0320 |
| **Target Deployment Environment** | **Edge Devices / Low-Resource CPUs** | High-End GPU Cloud Servers |

---

## 5. Visual Performance Plots

### Training & Convergence Curves
![Training Curves](file:///d:/Major%20Project/docs/figures/training_curves.png)

### Empirical Architectural Comparison
![Benchmark Comparison](file:///d:/Major%20Project/docs/figures/benchmark_comparison.png)

---

## 6. Evaluator Defense & Viva Q&A Guide

1. **Q: Why choose CNN + BiLSTM + Attention over pure Vision Transformers for the primary model?**  
   * **A:** Vision Transformers (like TrOCR) demand over 62 Million parameters and compute-intensive cross-attention decoding, resulting in severe latency and memory usage on CPU/edge devices. Our CNN + BiLSTM + Attention model achieves strong sequence modeling with only ~5.28M parameters (11.7x lighter), making it practical for real-time edge execution.

2. **Q: How does Self-Supervised Learning (SSL) mitigate labeled data dependency?**  
   * **A:** By applying contrastive learning (SimCLR InfoNCE loss) on unlabeled document and text crops, the CNN visual backbone learns robust spatial representations (stroke orientation, contours) before fine-tuning, dramatically lowering the volume of expensive hand-annotated labels required.

3. **Q: How does the system handle multi-line full-page documents?**  
   * **A:** The `LineSegmenter` module uses adaptive horizontal morphological dilation and projection profile clustering to segment multi-line documents into clean, isolated text strips with safe padding margins prior to sequence recognition.

4. **Q: How does your pipeline avoid deleting valid duplicate letters in CTC decoding?**  
   * **A:** Unlike naive argmax deduplication which stripped duplicate letters (e.g. `deep` $\rightarrow$ `dep`), our state-tracking CTC beam search only collapses identical consecutive character indices if no blank token ($0$) separates them, preserving correct spelling across English words and mathematical terms.

---

## 7. References (IEEE Format)
1. Y. Zhang et al., “Document Image Machine Translation,” in *Proc. ICDAR*, 2026.
2. C. Penarrubia, J. J. Valero-Mas, and J. Calvo-Zaragoza, “Self-Supervised Learning for Text Recognition: A Critical Survey,” *Int. J. Comput. Vis.*, 2025.
3. S. Sharma and A. Kumar, “Extraction of Text from Images Using Deep Learning,” *Procedia Computer Science*, vol. 235, pp. 100–108, 2024.
4. R. Mehta and P. Shah, “A Comparison Study on OCR Models in Mathematical Equation Recognition,” *Results in Control and Optimization*, vol. 18, 2025.
5. G. Kim et al., “OCR-Free Document Understanding Transformer,” *arXiv preprint arXiv:2111.15664*, 2022.
6. Y. Xu et al., “LayoutLMv2: Multi-Modal Pre-Training for Visually-Rich Document Understanding,” *arXiv preprint arXiv:2012.14740*, 2020.
7. M. Li et al., “TrOCR: Transformer-Based Optical Character Recognition with Pre-Trained Models,” *arXiv preprint arXiv:2109.10282*, 2022.
