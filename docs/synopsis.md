# 📄 MAJOR PROJECT WORK SYNOPSIS

**Institution:** N.M.A.M. Institute of Technology, Nitte – 574 110, Karnataka, India  
*(An Autonomous Institution affiliated to Visvesvaraya Technological University, Belagavi | ISO 9001:2015 Certified | Accredited with 'A' Grade by NAAC)*  
**Department:** Department of Information Science & Engineering  

---

## 👥 Student & Guide Information

| Student Name | USN | Signature |
| :--- | :--- | :--- |
| **Vishnumoorthy Raghavendra Bhat** | `NNM23IS215` | |
| **Tejaswi Shankar P** | `NNM23IS194` | |
| **Pranava Kumara K** | `NNM23IS132` | |
| **Simran Kasim Sayed** | `NNM23IS179` | |

* **Guide Name:** Dr. Rashmi Naveen
* **Project Code:** 42
* **Project Title:** **Efficient Text Recognition from Images Using Deep Learning and Self Supervised Learning**

---

## 1. Introduction
Text recognition from images is a fundamental problem in computer vision and pattern recognition, playing a vital role in applications such as document digitization, automatic form processing, assistive technologies, and scene text understanding. Traditional Optical Character Recognition (OCR) systems rely heavily on handcrafted features and rule-based techniques, which often fail when dealing with noisy images, complex backgrounds, or handwritten text.

Recent advances in deep learning have significantly improved OCR performance by enabling automatic feature extraction and sequence modeling. However, most deep learning–based approaches require large amounts of labeled training data and high computational resources, making them difficult to deploy in low-resource environments.

To address these challenges, this project explores an efficient text recognition framework that integrates deep learning with self-supervised learning techniques. The aim is to reduce dependency on labeled data, improve recognition accuracy for complex text, and optimize computational efficiency.

---

## 2. Background and Context
With the increasing availability of digital images and scanned documents, there is a growing demand for accurate and robust text recognition systems. While deep neural networks have achieved impressive results, their dependence on large annotated datasets limits scalability and real-world applicability.

Self-supervised learning has emerged as a promising approach that allows models to learn useful representations from unlabeled data. By leveraging this paradigm, it becomes possible to train OCR systems with fewer labeled samples while maintaining high performance.

This project is motivated by the need to build a practical and efficient text recognition system that combines deep learning and self-supervised learning to overcome existing limitations in traditional OCR and modern deep-learning-based methods.

---

## 3. Scope and Limitations

### Scope
- Design and implementation of a deep learning–based text recognition system.
- Integration of self-supervised pretraining (SimCLR / InfoNCE contrastive representation).
- Recognition of printed and handwritten text, mathematical equations, and historical scripts.
- Performance evaluation using standard metrics (CER, WER, Latency, Parameters Count, Throughput FPS).

### Limitations
- Limited training time and computational resources.
- Dataset size may restrict generalization.
- Focus is on English text and standard mathematical notations only.
- Extremely low-quality / heavily degraded images may still cause recognition errors.

---

## 4. Objectives of the Project
- **To implement OCR system using CNN and transformer-based methods and compare the performance.**
- **To improve recognition accuracy for handwritten and complex text.**
- **To develop an optimized OCR system with lower computational complexity.**
- **To reduce dependency on large labeled datasets using self-supervised learning techniques.**

---

## 5. Problem Statement
Despite advancements in OCR technology, existing systems still struggle with handwritten text, noisy images, and limited labeled data availability. Furthermore, high computational requirements restrict deployment on low-resource devices. Therefore, there is a need for an efficient text recognition system that achieves high accuracy while reducing labeling effort and computational cost.

---

## 6. Methodology
1. **Data Collection and Preprocessing:** Collect and preprocess printed, handwritten, and mathematical image datasets with aspect-ratio preservation, foreground deskewing, and adaptive contrast enhancement (CLAHE + Bilateral filtering).
2. **Self-Supervised Pretraining:** Perform self-supervised contrastive representation learning (SimCLR) on unlabeled text crops.
3. **Supervised Architecture Fine-Tuning:** Fine-tune deep learning sequence models (`CNN + BiLSTM + Attention`) with Connectionist Temporal Classification (CTC) loss.
4. **Baseline Vision Transformer Benchmark:** Implement and compare with pre-trained Vision Transformer (`TrOCR`) encoder-decoder models.
5. **Text Recognition & Linguistic Post-Processing:** Perform CTC prefix beam search sequence decoding, OCR confusion matrix corrections, split-word repairing, and LaTeX formula protection.
6. **System Performance Evaluation:** Evaluate recognition performance across Character Error Rate (CER), Word Error Rate (WER), latency (ms), and parameter efficiency.

---

## 7. Significance and Expected Outcomes
This project is expected to produce a robust OCR system that performs well with limited labeled data. The integration of self-supervised learning will reduce annotation effort and improve generalization. The optimized model will be suitable for real-world applications such as document digitization, educational tools, and assistive technologies.

---

## 8. Conclusion
The proposed project aims to enhance text recognition performance by implementing and comparing CNN-based and transformer-based OCR methods. By addressing the challenges of accuracy on handwritten and complex text as well as computational efficiency, the system is expected to provide a robust and practical solution for modern OCR applications. Future work may extend the system to multilingual text recognition and real-time deployment.

---

## 9. References (IEEE Style Format)
1. Y. Zhang et al., “Document Image Machine Translation,” in *Proc. ICDAR*, 2026.
2. C. Penarrubia, J. J. Valero-Mas, and J. Calvo-Zaragoza, “Self-Supervised Learning for Text Recognition: A Critical Survey,” *Int. J. Comput. Vis.*, 2025.
3. S. Sharma and A. Kumar, “Extraction of Text from Images Using Deep Learning,” *Procedia Computer Science*, vol. 235, pp. 100–108, 2024.
4. R. Mehta and P. Shah, “A Comparison Study on OCR Models in Mathematical Equation Recognition,” *Results in Control and Optimization*, vol. 18, 2025.
5. G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, and S. Park, “OCR-Free Document Understanding Transformer,” *arXiv preprint arXiv:2111.15664*, 2022.
6. Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, and L. Zhou, “LayoutLMv2: Multi-Modal Pre-Training for Visually-Rich Document Understanding,” *arXiv preprint arXiv:2012.14740*, 2020.
7. M. Li, T. Lv, J. Chen, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, “TrOCR: Transformer-Based Optical Character Recognition with Pre-Trained Models,” *arXiv preprint arXiv:2109.10282*, 2022.
