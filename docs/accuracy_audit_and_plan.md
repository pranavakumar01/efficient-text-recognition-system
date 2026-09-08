# Accuracy Audit and Remediation Plan

**Project:** Efficient Text Recognition System (DL + SSL)
**Audit date:** 2026-09-01
**Scope:** Root-cause analysis of the reported CER of 0.5348 on the CNN + BiLSTM + Attention model, and of the CER of 1.032 on the TrOCR baseline.
**Method:** Static reading of the training, dataset, model, inference, preprocessing, and evaluation code, plus inspection of the annotation files and the installed environment. No code was executed; items marked *needs runtime confirmation* are hypotheses that require a run on the training machine to settle.

---

## 1. Executive summary

The reported error rates are not primarily a modelling-capacity problem. They are the compound result of four independent defects, any one of which would be enough to produce a badly degraded score:

The model is trained on one image distribution and evaluated on a different one, because the `ImagePreprocessor` chain runs at inference but not during training. Fifty-three percent of the training set is handwritten LaTeX mathematics, a task that the CTC objective is structurally incapable of representing. The Bahdanau attention block that the project's title and report present as the primary contribution computes a context vector and then discards it before the classifier, so the trained network is functionally a plain CRNN. And the SimCLR pretraining that constitutes the project's third headline feature uses an encoder that global-average-pools to a single spatial cell, destroying exactly the per-column sequence information the downstream recogniser needs.

Separately, the evaluation set is a subset of the training set, so the 0.5348 figure is optimistically biased. The honest number, once splits are made disjoint, will initially be worse than what is currently reported. That is the correct direction of travel: the present figure is both bad *and* unearned.

A realistic target after remediation, for single-line printed and handwritten word recognition with this architecture and roughly 1,800 non-math samples, is a CER in the range of 0.05 to 0.15. Reaching the lower end of that band will require more data than the project currently has.

---

## 2. Ranked remediation table

Impact is the expected reduction in CER. Effort is engineering time, not compute.

| # | Defect | Impact | Effort | Class |
|---|---|---|---|---|
| 1 | Train/inference preprocessing mismatch | High | Low | Bug |
| 2 | LaTeX math (53% of data) is unlearnable under CTC | High | Low | Design error |
| 3 | Evaluation set overlaps the training set | Corrects bias | Medium | Methodology |
| 4 | Augmentation implemented but never enabled | Medium | Trivial | Bug |
| 5 | `input_lengths` ignores per-sample true width | Medium | Low | Bug |
| 6 | Attention block is a no-op; context vector discarded | Medium | Medium | Design error |
| 7 | Beam-search normalised score used for per-timestep pruning | Low–Medium | Low | Bug |
| 8 | Only 5 training epochs | Medium | Trivial | Tuning |
| 9 | SSL encoder is 2 stages and global-pools away sequence info | Low direct | High | Design error |
| 10 | 95-character vocabulary far exceeds label charset | Low–Medium | Low | Tuning |
| 11 | Checkpoint selected on val loss rather than val CER | Low | Trivial | Tuning |
| 12 | No input standardisation beyond division by 255 | Low | Trivial | Tuning |
| 13 | Fabricated placeholder predictions on empty output | Reporting | Trivial | Integrity |
| 14 | EasyOCR can silently replace the CNN's prediction | Reporting | Trivial | Integrity |
| 15 | TrOCR processor forced to 384×384, destroying aspect ratio | High (baseline) | Low | Bug |
| 16 | TrOCR parameter count hardcoded to 62M | Reporting | Trivial | Integrity |

---

## 3. Tier 1 — defects that dominate the error rate

### 3.1 Train and inference see different image distributions

`OCRDataset.preprocess_image` in `src/dataset.py` performs exactly three operations: it reads the file as greyscale, resizes it to height 32 preserving aspect ratio with `INTER_AREA`, and divides by 255.

`OCRInferenceEngine.predict_cnn` in `src/infer.py` instead calls `ImagePreprocessor.process`, whose `final` output is the product of bilateral denoising, CLAHE contrast equalisation with `clipLimit=2.5`, Otsu-based deskewing, an aspect-preserving resize that is capped at 640 pixels of width, and white padding out to a multiple of 16.

The network is therefore asked at test time to read images whose local contrast statistics, stroke thickness, and skew have all been altered by transforms it never encountered during fitting. CLAHE in particular rewrites the intensity histogram tile by tile, which is precisely the signal a shallow CNN's early filters key on. The 640-pixel width cap adds a second mismatch: long lines are horizontally compressed at inference but were not during training, so the effective character pitch differs.

There must be exactly one preprocessing path. The preferable direction is to move `ImagePreprocessor` into the dataset so that training sees the same enhanced, deskewed, padded images that inference produces, since deskewing and contrast normalisation are genuinely useful for handwritten input. The alternative — stripping the preprocessor at inference — is cheaper but discards real robustness.

Whichever direction is chosen, `resize_and_pad`'s width cap should be applied identically in both paths, and the padding-to-multiple-of-16 step should be retained in both so that the timestep count is predictable.

### 3.2 CTC cannot represent LaTeX mathematics

`data/mathwriting/annotations.csv` contributes 2,000 of the 3,790 total samples. Its labels are LaTeX source strings such as `\dot{y}=\frac{dy}{dt}`, `\phi^{*}(T^{*}N)\rightarrow T^{*}M`, and `{10^{218}}^{10}-\frac{\frac{4}{\sqrt{375}}}{179}`.

CTC's central assumption is a monotonic, order-preserving alignment between input timesteps and output symbols: symbol *k* must be emitted at a timestep no earlier than symbol *k−1*. LaTeX for two-dimensional layout violates this. In `\frac{dy}{dt}` the numerator and the denominator occupy the *same* horizontal band of image columns, so the token sequence `d y } { d t` cannot be laid against left-to-right image columns in any monotonic way. Superscripts, nested radicals, and matrix environments have the same property. No amount of training will fix this, because the target function is not in the hypothesis class that CTC can express.

There is a second, mechanical failure on top of the structural one. Tracing `CNNFeatureExtractor`, an input of height 32 passes `pool1` (2,2), `pool2` (2,2), and `pool3` with kernel and stride (2,1), reaching height 4, at which point `conv5` with kernel (4,1) and no padding collapses height to exactly 1. Width is halved twice and then left unchanged, so the number of timesteps is **T = floor(W/4)**. The floor matters slightly: `ocr_collate_fn` pads to the batch maximum width with no constraint that the result be divisible by four, so up to three columns of the rightmost image content are silently discarded. Because width is set by aspect-preserving resize to height 32, a math crop of 300×60 pixels yields W = 160 and therefore T = 40. The label `{10^{218}}^{10}-\frac{\frac{4}{\sqrt{375}}}{179}` is 46 characters long. Since T < L, no valid CTC path exists and the loss is infinite. `nn.CTCLoss` is constructed with `zero_infinity=True`, so these samples are silently assigned zero loss and zero gradient, with no warning emitted.

The practical consequence is that a large fraction of the math data contributes no learning signal at all while still having shaped the tokenizer's 95-character vocabulary and, through the samples that *do* fit, having pulled the classifier's output prior heavily toward `\`, `{`, `}`, `^`, and `_`. The model is being taught to expect LaTeX punctuation in images of printed English sentences.

The fix is to exclude `data/mathwriting` from CTC training. The project need not abandon its mathematics claim: the defensible framing is that CTC is appropriate for single-line text and structurally unsuited to two-dimensional mathematical layout, and that math recognition is therefore addressed by a separate attention-based encoder–decoder head, or scoped as future work. Stating this explicitly is a stronger contribution than a silently broken joint model, because it demonstrates that the limitation is understood rather than unnoticed.

### 3.3 The evaluation set is inside the training set

`train_model` builds its corpus from four directories — the `data_dir` argument plus hardcoded paths to `mathwriting`, `expanded`, and `kaggle_dataset` — and constructs each `OCRDataset` **without passing `split`**. It then applies `random_split` at 85/15 to the concatenation. Meanwhile `run_evaluation` defaults to `data_dir="d:\Major Project\data\expanded"` with `split=None`.

`data/expanded/annotations.csv` has no `split` column at all; its header declares `image_path,label,category` and its rows supply only the first two fields. `data/kaggle_dataset/annotations.csv` likewise has no split column. Only `data/mathwriting/annotations.csv` carries one.

So all 600 `expanded` samples enter training, and all 600 are then used for the benchmark. The reported 0.5348 is a partially-memorised training score. The label set makes this worse: the `expanded` labels are drawn from a small pool of repeated template strings — `Research Meeting Notes on Transformer Models` appears at indices 0005 and 0009, and `sigma(z) = 1 / (1 + e^(-z))` at 0007, 0010, and 0012. With only a few dozen distinct target strings, all seen in training, a functioning model should approach a CER near zero by memorisation alone. Scoring 0.5348 under those conditions is strong evidence of a genuine pipeline failure rather than a data-scarcity problem.

Remediation requires writing a `split` column into every annotation file, partitioning by *distinct label string* rather than by row so that the same template text cannot appear on both sides, and passing `split="train"` in training and `split="test"` in evaluation.

One complication to note before starting: no held-out partition currently exists anywhere in the project. `mathwriting` is the only source with a `split` column, and every image filename in it carries the `train_` prefix, so even that column contains a single value. The splits must be created from scratch rather than recovered.

### 3.4 Augmentation never runs

`OCRDataset.__init__` accepts `augment: bool = False`. Every construction site — the four in `train_model`, the one in `run_evaluation` — omits the argument. `augment_image`, with its rotation, blur, and brightness jitter, is dead code.

Enabling it for the training split only is a one-line change with a real payoff on the handwritten subset, where the 1,040 Kaggle samples are photographs of handwritten French surnames with natural variation in slant and stroke weight. Augmentation must remain off for validation and test.

### 3.5 CTC is told every sample fills the padded width

`ocr_collate_fn` pads all images in a batch to the batch's maximum width using white pixels. `train_model` then sets

```python
input_lengths = torch.full(size=(b_size,), fill_value=seq_len, ...)
```

so every sample claims the full padded timestep count. A four-character word like `PAUL` padded into a batch alongside a 42-character sentence is declared to have the same number of valid timesteps as the sentence. The model must learn to emit blanks across a long stretch of white padding, and the per-sample loss normalisation is distorted.

The collate function should return the true pre-padding widths, and training should compute `input_lengths` as the per-sample width divided by four, matching the CNN's exact downsampling factor. Bucketing the sampler by width would additionally cut wasted computation and reduce the padding the model has to explain away.

---

## 4. Tier 2 — architectural defects

### 4.1 The attention mechanism does nothing

This is the most consequential finding for the report, as distinct from the error rate. The forward pass reads:

```python
query = torch.mean(lstm_out, dim=1)              # [B, 2H] — one query for the whole sequence
context, att_weights = self.attention(query, lstm_out)
attended_seq = lstm_out * (1.0 + att_weights)    # [B, T, 2H] * [B, T, 1]
logits = self.classifier(attended_seq)
```

Three things are wrong. `context` — the actual output of Bahdanau attention, the weighted sum of values computed at line 59 — is bound at line 132 and never read again; `attended_seq` and `logits` depend only on `att_weights`, so the entire context-vector computation is work thrown away. The modulation that *is* applied uses `att_weights`, which `BahdanauAttention` produces through `self.V`, an `nn.Linear(hidden_dim, 1)`, followed by `F.softmax(score, dim=1)` over the time axis. The weights therefore sum to one across T, which bounds the multiplier `1.0 + att_weights` within [1.0, 2.0] with a mean of 1 + 1/T. This is a weak, non-negative multiplicative reweighting of the sequence rather than an attention mechanism in any meaningful sense: it can rescale a few timesteps by at most a factor of two and leaves the rest essentially untouched, and it cannot introduce information from one timestep into another, which is the entire purpose of attention. And because the query is a single mean-pooled vector rather than one query per position, the mechanism cannot in principle produce the per-timestep context that a CTC classifier consumes.

The model that produced the checkpoint is, functionally, a plain CRNN — CNN, BiLSTM, CTC — with three unused `nn.Linear` layers attached. The report, README, and slide deck all present "CNN + BiLSTM + Bahdanau Attention Decoder" as the primary architectural contribution. An examiner who reads `forward()` will see that the attention output is discarded, and that is a difficult question to field.

The repair that preserves the claim honestly is to replace the single-query Bahdanau block with per-timestep self-attention over the BiLSTM output: project the LSTM states to queries, keys, and values, compute scaled dot-product attention so that each timestep attends over the full sequence, and add the result back to the LSTM output through a residual connection and layer normalisation before the classifier. This keeps the timestep count intact — which CTC requires — gives every position a genuine context vector, and is a real, describable mechanism. It is also a modest amount of code.

If a Bahdanau formulation specifically must be retained for continuity with the literature survey, the equivalent is to iterate the existing `BahdanauAttention` with `query = lstm_out[:, t, :]` for each t and stack the resulting contexts, which is the additive-attention analogue of the same idea at higher computational cost.

### 4.2 The SSL backbone cannot transfer, and learns the wrong kind of feature

`load_ssl_weights` maps twelve tensors: weights, biases, and BatchNorm statistics for `conv1`/`bn1` and `conv2`/`bn2`. The supervised feature extractor has five convolutional stages, so two of five receive pretrained initialisation. Stages three, four, and five hold the overwhelming majority of the 5.28M parameters, as channel counts rise through 256 and 512, and they stay randomly initialised.

The reason the mapping stops at two stages is not an oversight in the mapping — it is that `SimCLR_SSL_Backbone` in `src/models/ssl_backbone.py` has only two convolutional stages to give. Its `encoder` is a six-layer `nn.Sequential` comprising `Conv2d(1,64)`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `Conv2d(64,128)`, `BatchNorm2d`, `ReLU`, and then `AdaptiveAvgPool2d((1,1))`. It is a different and far shallower architecture than `CNNFeatureExtractor`, so no mapping could transfer more than it does.

The `AdaptiveAvgPool2d((1,1))` is the deeper problem. It collapses the entire feature map to a single spatial cell, so the contrastive objective is trained on a global 128-dimensional descriptor of the whole crop. InfoNCE then teaches the encoder to make that global descriptor invariant to the augmentations applied to the two views. But a sequence recogniser needs the opposite: features that vary sharply along the width axis, because position-specific stroke evidence is precisely what distinguishes one character column from the next. The pretraining objective is therefore not merely under-transferred, it is pointed away from the downstream task. Whatever the two transferred stages learned, they learned it in service of discarding spatial layout.

The fix is structural and has two parts. First, have the SSL module instantiate and use the *same* `CNNFeatureExtractor` class as the supervised model, so that transfer is a single `load_state_dict` on a shared submodule and the whole backbone carries over. Second, replace the global average pool with a pooling scheme that preserves the width axis — average over height only, leaving a sequence of column descriptors — and apply the contrastive loss over those column features, or over crops taken at matched horizontal positions in the two views. Augmentations must then be chosen so they do not translate the image horizontally, since horizontal position is now the signal being preserved rather than the nuisance being removed.

Only after both parts are done will the SSL ablation in the report measure anything. As the code currently stands, the honest reporting of `--use_ssl` is that it initialises the two cheapest layers from a global-descriptor objective, and that no effect on CER should be expected. Note also that the mapping silently skips keys absent from the checkpoint and returns `True` regardless, so a layout mismatch reports success with `loaded_count = 0`; and `num_batches_tracked` is not in the mapping, so the transferred BatchNorm statistics are partially reset.

### 4.3 The beam search prunes on a length-normalised score

An earlier draft of this audit claimed that the decoder's length normalisation had an inverted sign and systematically rewarded longer hypotheses. **That claim was wrong and has been withdrawn.** It is recorded here because the reasoning is a tempting trap. Beams are ranked by

```python
np.logaddexp(p_b, p_nb) / (max(1, len(prefix)) ** 0.65)
```

and it is true that the numerator is a negative log-probability, so dividing by a denominator greater than one moves it toward zero. But the numerator is not held fixed as length varies: extending a prefix adds a non-positive term at every step, so a length-L prefix accumulates a numerator on the order of −cL, and the normalised score behaves like −c·L^0.35, which grows *more* negative as L increases. With the exponent 0.65 being less than one, the normalisation attenuates CTC beam search's intrinsic bias toward short prefixes without inverting it. This is standard GNMT-style length normalisation, not a defect.

The real defect in this function is narrower. The normalised score is used not only for final beam selection but also for per-timestep pruning, at the `sorted_beams` call inside the timestep loop. At that point the surviving beams have unequal lengths, so pruning compares scores that have been divided by different denominators, which distorts which beams survive in a way that final-selection normalisation does not. The normalisation should be applied only when choosing among completed beams; pruning inside the loop should compare raw accumulated log-probabilities. Note also that `max(1, len(prefix))` gives the empty prefix and every length-one prefix the same divisor, and that the comment describing this as a "length penalty" is a misnomer, since relative to raw log-probability it is a length bonus.

Two further points on the same function. The inner loop iterates over all 96 classes for every beam at every timestep in pure Python, which accounts for much of the 624 ms latency; restricting the loop to the top-k classes per timestep, with k around 8, is numerically almost lossless and dramatically faster. And the `lm_bonus` of 0.8 is added inside the log-domain accumulation at space boundaries, which conflates a language-model prior with the recognition score in a way that is difficult to justify or tune; it should be applied once per completed word at rescoring time, if at all. That bonus draws on `OCRPostProcessor.DOMAIN_SET`, which as discussed in section 6 is a list of roughly 95 project-specific jargon words rather than a general dictionary, so it rewards the decoder for producing words from the report's own vocabulary.

### 4.4 The vocabulary is larger than the data supports

`DEFAULT_VOCAB` is the full printable ASCII set: 10 digits, 52 letters, and 33 punctuation and symbol characters, giving 95 classes plus the CTC blank for 96 outputs. Once the LaTeX subset is removed, the remaining labels are printed English phrases and uppercase surnames, whose combined charset is far smaller — letters, digits, space, and a handful of punctuation marks.

A 96-way softmax trained on roughly 1,800 samples spreads the classifier's capacity across classes that never appear in a target. Building the vocabulary from the observed training labels, and asserting at load time that no label contains an out-of-vocabulary character, is straightforward and recovers some capacity.

That assertion matters for a second reason. `Tokenizer.encode` currently skips unknown characters silently:

```python
for char in text:
    if char in self.char2idx:
        encoded.append(self.char2idx[char])
```

Any label character outside the vocabulary is dropped from the CTC target while remaining in the ground-truth string that CER is computed against. The model is optimised toward a shortened target and then scored against the full one, which inflates CER through no fault of the network. Silent truncation should become a loud failure.

---

## 5. Tier 3 — training recipe

The argparse default is `--epochs 5`, though the README documents 15. With roughly 3,200 training samples at batch size 8, five epochs is about 2,000 optimiser steps. Convolutional-recurrent OCR models are normally trained for tens of thousands of steps; CTC in particular spends its early phase collapsed onto the blank class before alignments sharpen, and 2,000 steps may not clear that phase. Training should run for at least 100 epochs with early stopping keyed to validation CER.

The best checkpoint is currently selected on validation loss. CTC loss and CER are only loosely coupled, and a model can trade loss for alignment quality in either direction. Select on validation CER, which is the quantity being reported.

A learning rate of 1e-3 under AdamW is on the hot side for CTC trained from random initialisation and is a common cause of blank collapse. A rate of 3e-4 with a few hundred warmup steps, retaining the existing cosine decay to 1e-5, is a safer starting point.

Inputs are scaled to [0,1] by division by 255 with no further standardisation. Normalising to roughly zero mean and unit variance, for instance by subtracting 0.5 and dividing by 0.5, conditions the first convolution better. Whatever constants are chosen must be applied identically in training and inference, which follows automatically once section 3.1 is resolved.

---

## 6. Reporting integrity

These items do not change what the model has learned, but they determine whether the reported numbers describe the model. They should be settled before any figure is written into the report.

`decode_predictions` in `src/infer.py` falls back to `raw_text = "Recognized Text Sample"` when the model emits nothing, and `TrOCRBaseline.predict` falls back to `"Recognized Text Extraction"`. Both substitute an invented transcription for a real failure, and both are scored as though they were predictions. They should return the empty string, and empty predictions should be counted honestly as complete misses.

More seriously, `decode_predictions` contains a live EasyOCR substitution path:

```python
if (not raw_text or len(raw_text) < 3 or not has_valid_words) and image_np is not None and self.easy_reader is not None:
    res = self.easy_reader.readtext(image_np)
    ...
    raw_text = easy_text
```

The trigger includes `not has_valid_words`, which is true whenever the CNN's output contains no token from `OCRPostProcessor.DOMAIN_SET`. That set is not a dictionary: it is a hardcoded list of roughly 95 project-specific jargon words — `bilstm`, `trocr`, `nmamit`, `bahdanau`, `quick`, `brown`, `fox`, and similar — defined in `src/utils/postprocessing.py`. The `pyspellchecker` instance available in that module is not consulted here. So the substitution fires unless the model happens to emit one of those 95 strings, which for a weak model is the overwhelming majority of cases. The condition is broader still, because `has_valid_words` can only become `True` when the output is at least four characters long, so every three-character prediction triggers the swap regardless of content.

`easyocr` version 1.7.2 is present in the virtual environment, so `self.easy_reader` is expected to be non-None and this branch is live. The consequence is that the value reported under `"model_name": "CNN + BiLSTM + Attention (TTA)"` is, for most inputs, EasyOCR's transcription rather than the project's model. Whether it was so on the specific run that produced the current CSV *needs runtime confirmation*, but the number cannot be attributed to the CNN until the path is removed. It must be deleted from the evaluation path entirely. If a third-party recogniser is wanted in the web demo as a convenience, it belongs there as a separate, clearly labelled model option, never as a silent fallback inside the primary model's own prediction function. This is the single most important item in this document to fix, because unlike the accuracy defects it affects whether the reported result describes the work at all.

Two further points affect what the CNN's CER measures. `predict_cnn` defaults to `use_tta=True`, running three forward passes over differently-enhanced copies of the image and averaging the probabilities; the reported latency of 624 ms is consequently for a three-member ensemble, which weakens the efficiency argument the project is built on. And CER is computed on the output of `OCRPostProcessor.process` with autocorrect enabled, so the figure describes the model plus a dictionary corrector. Both are legitimate configurations, but the report should give the raw single-pass, no-autocorrect number as the model's score and present TTA and autocorrect as separately-quantified additions.

Finally, the baseline row. `evaluate.py` hardcodes `trocr_params = 62000000`, while `TrOCRBaseline.__init__` defaults to `microsoft/trocr-base-printed`, which is roughly 334M parameters; the 62M figure corresponds to `trocr-small-printed`, the fallback and the model the README names. The parameter count should be measured from the loaded model, not asserted.

---

## 7. Why the TrOCR baseline scores above 1.0

A CER above 1.0 means the edit distance exceeds the reference length, which requires the hypothesis to be substantially longer than the reference. The most likely mechanism is in `_load_model`:

```python
img_p = ViTImageProcessor.from_pretrained(
    self.model_name,
    size={"height": 384, "width": 384},
    do_resize=True, ...
)
```

Overriding the processor to a square 384×384 forces every input through a non-aspect-preserving resize. A 500×50 text line is squashed to a square, rendering the glyphs unreadably compressed vertically and stretched horizontally. TrOCR's own pretrained configuration handles line images correctly; overriding it discards that. Fed unreadable input and permitted `max_new_tokens=64` with four-beam search, an autoregressive decoder characteristically rambles, producing long fluent strings unrelated to the image. Long hypothesis against a 35-character reference gives a CER above 1.0 and a WER above 1.0, which matches the reported 1.032 and 1.1417.

Supporting evidence that the transformer genuinely ran, rather than falling through to a placeholder: the measured latency is 5,627 ms per image, which is consistent with real beam-search generation from a 334M-parameter encoder–decoder on CPU, and inconsistent with returning a constant string.

The fix is to construct the processor with `TrOCRProcessor.from_pretrained(self.model_name)` and no size override, decide deliberately between `trocr-base-printed` and `trocr-small-printed` and use the same one consistently across code, README, and report, and measure the parameter count. Note also that `transformers` in this environment is version 5.14.1, a major release in which several of the imported classes have been renamed or removed; whether `ViTImageProcessor`, `DeiTImageProcessor`, and the keyword form `TrOCRProcessor(image_processor=..., tokenizer=...)` still resolve on 5.14.1 *needs runtime confirmation*. If they do not, the except branch is being taken and the diagnosis above would need revisiting.

Once the baseline is configured correctly it will become genuinely strong on printed text, with a CER plausibly in the 0.05 to 0.15 range. The project's comparative argument must then rest on efficiency — roughly 5.3M parameters against 334M, and single-pass latency — rather than on raw accuracy. That is a defensible and more interesting thesis than the current one, and it is the thesis the benchmark table was presumably meant to support.

---

## 8. Suggested order of work

The dependencies matter, because measuring anything before the splits are honest wastes the measurement.

Begin by writing disjoint `split` columns into all annotation files, partitioned by distinct label string, and excluding `mathwriting` from the CTC corpus. Then unify the preprocessing path so training and inference share one function. Then correct the mechanical bugs: enable augmentation on the training split only, compute true per-sample `input_lengths`, remove the EasyOCR fallback and the fabricated placeholder strings from the evaluation path, and move the beam search's length normalisation out of the per-timestep pruning step.

With that in place, retrain and record a clean baseline number. It will be worse than 0.5348 and it will be the first trustworthy figure the project has produced.

Only then change the architecture: replace the no-op attention with per-timestep self-attention, restructure the SSL module to share `CNNFeatureExtractor` and to pool over height only so that column features survive, and rebuild the vocabulary from observed labels. Retrain with the longer schedule, the lower learning rate, and checkpoint selection on validation CER.

Finally, repair the TrOCR configuration and re-run the comparison, reporting measured parameter counts and single-pass latencies for both models.

Each stage should produce a recorded CER on the held-out test split, so that the report can present a genuine ablation showing what each change contributed. That table — showing the effect of preprocessing alignment, of removing the math corpus, of real attention, and of SSL transfer — is a considerably stronger result than a single number, and it comes for free if the numbers are recorded as the work proceeds.
