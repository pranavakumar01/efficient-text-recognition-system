# 📊 Presentation-1 Slide Deck & Speaker Notes (Aug 7)
**Project Code:** 42  
**Title:** *Efficient Text Recognition from Images Using Deep Learning and Self Supervised Learning*  
**Department:** Information Science & Engineering, NMAMIT Nitte  
**Guide:** Dr. Rashmi Naveen  
**Team Members:** Vishnumoorthy R Bhat, Tejaswi Shankar P, Pranava Kumara K, Simran Kasim Sayed  

---

## Slide 1: Title & Team Credentials
* **Project Code:** 42
* **Project Title:** Efficient Text Recognition from Images Using Deep Learning and Self Supervised Learning
* **Team:**
  * Vishnumoorthy R Bhat (NNM23IS215)
  * Tejaswi Shankar P (NNM23IS194)
  * Pranava Kumara K (NNM23IS132)
  * Simran Kasim Sayed (NNM23IS179)
* **Guide:** Dr. Rashmi Naveen, Assistant Professor, Dept. of ISE

---

## Slide 2: Problem Identification & Motivation (10 Marks Focus)
* **High Labeling Dependency:** Traditional & supervised deep learning OCR models require massive labeled datasets, which are expensive and time-consuming to create.
* **Degraded & Complex Text Challenges:** Scene text, mathematical formulas, handwritten notes, and historical documents suffer from noise, curvature, and occlusion.
* **Deployment Constraints:** Modern Vision Transformers (e.g., TrOCR) have massive parameter counts (>60M params) and high latency, making them impractical for real-time or edge device deployment.
* **Motivation:** Build a **label-efficient, lightweight OCR framework** integrating **Self-Supervised Learning (SSL)** with a **CNN + BiLSTM + Attention** pipeline for low-resource environments.

---

## Slide 3: Literature Survey & Identified Gaps (5 Marks Focus)
* **Reviewed Papers:** Analyzed 40 papers spanning CNN-RNN models, Vision Transformers, and SSL techniques (SimCLR, MAE-OCR, TrOCR).
* **Gap 1:** Lack of fully integrated end-to-end SSL OCR pipelines combining visual representation and sequence modeling.
* **Gap 2:** Limited evaluation of SSL algorithms on real-world noisy, handwritten, and historical text images.
* **Gap 3:** High computational overhead in Transformer-based architectures without latency optimization.

---

## Slide 4: Project Objectives
1. **Model Implementation & Comparison:** Implement a **CNN + BiLSTM + Attention** OCR system and compare its performance directly against a **Vision Transformer (TrOCR)** baseline. *(Guide Recommendation)*
2. **Data Efficiency:** Reduce labeled data dependency using Self-Supervised pre-training (SimCLR contrastive learning).
3. **Robustness:** Enhance recognition accuracy across printed, handwritten, mathematical, and historical document text.
4. **Computational Optimization:** Achieve high inference speed and low parameter footprint suitable for edge deployment.

---

## Slide 5: Methodology & System Architecture (5 Marks Focus)
```text
[Raw Image Input]
       ↓
[Preprocessing Engine] (Grayscale, CLAHE Contrast, Denoising, Adaptive Binarization, Deskewing)
       ↓
[SSL Representation Encoder] (SimCLR / InfoNCE Pretrained Feature Representation)
       ↓
┌───────────────────────────────────────┬──────────────────────────────────────┐
│ Primary Model (CNN + BiLSTM + Attn)  │ Baseline Comparison (TrOCR / ViT)   │
│  • ResNet Backbone Feature Extraction │  • Vision Encoder Decoder Transformer│
│  • Bidirectional LSTM Sequence Model  │  • Generative Text Decoder           │
│  • Bahdanau Attention Mechanism       │                                      │
└───────────────────────────────────────┴──────────────────────────────────────┘
       ↓
[Interactive Web Application & Metrics Engine] (CER, WER, Latency ms, Parameters Count)
```

---

## Slide 6: Preliminary Implementation & Web Demo Progress
* **Preprocessing Pipeline:** Implemented OpenCV adaptive contrast enhancement & deskewing.
* **Model Codebase:** Built modular PyTorch implementation of `CNN + BiLSTM + Attention` and `Vision Transformer` wrapper.
* **Web UI Dashboard:** Developed interactive FastAPI web app displaying visual preprocessing stages and real-time comparative benchmarks.

---

## Slide 7: Phase-I Timeline & Future Work
* **Aug 7:** Presentation-1 Defense.
* **Aug 14 - Aug 28:** Demo-1 Functional Module Evaluation.
* **Sept - Oct:** Dataset fine-tuning (IAM / Synthetic math dataset), SSL pre-training optimization, and final report preparation.

---

## ❓ Frequently Asked Evaluator Questions & Answers (Q&A Defense)

1. **Q: Why did you choose CNN + BiLSTM + Attention over pure Vision Transformers?**
   * **A:** Vision Transformers offer high accuracy but require tens of millions of parameters (~62M+) and high latency. As per our guide's recommendation, the CNN + BiLSTM + Attention architecture provides a highly efficient balance of low parameter footprint, low latency (suitable for edge deployment), and strong sequence modeling.

2. **Q: How does Self-Supervised Learning (SSL) help in this project?**
   * **A:** SSL allows our feature encoder to learn rich visual representations from unlabeled text crops (using contrastive loss like SimCLR / InfoNCE), drastically reducing the requirement for expensive human-annotated ground truth data.

3. **Q: How do you handle non-standard text like mathematical equations or handwritten text?**
   * **A:** Our preprocessing pipeline uses adaptive binarization and CLAHE contrast enhancement to isolate stroke features, while the BiLSTM + Attention layers model arbitrary character sequences and symbols effectively.
