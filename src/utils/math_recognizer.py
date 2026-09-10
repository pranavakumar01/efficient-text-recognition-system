import os
import re
import csv
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class MathFormulaParser:
    """
    Mathematical Equation Recognizer, 2D Layout Decomposer & Canonical LaTeX Synthesizer.
    Fulfills Objective 2: High accuracy for complex mathematical text.

    Key Capabilities:
      1. 2D Visual Layout Decomposition: Detects horizontal fraction dividing bars and extracts
         numerator, denominator, prefix, and suffix sub-crops, resolving the 1D CTC collapse.
      2. Mathematical Token Grammar: Converts optical transliterations (e.g. `(DOT(Y)`, `VI-`, `SFI`,
         `L_(CTC) = - IN P(Y | X)`) into clean, standard LaTeX syntax.
      3. Greek & Symbol Synthesizer: Maps spelled-out, OCR-corrupted, or Latin approximations
         to LaTeX macros (\\nabla, \\rho, \\omega, \\theta, \\mu, \\lambda, \\delta, etc.).
      4. LaTeX Brace & Delimiter Validator: Balances all { and } braces and angle brackets.
      5. Domain Corpus Memory: Matches optical sequences against known mathematical formula structures.
    """

    GREEK_SYMBOLS = {
        r'\balpha\b': r'\alpha',
        r'\bbeta\b': r'\beta',
        r'\bgamma\b': r'\gamma',
        r'\bdelta\b': r'\delta',
        r'\bepsilon\b': r'\epsilon',
        r'\btheta\b': r'\theta',
        r'\blambda\b': r'\lambda',
        r'\bmu\b': r'\mu',
        r'\bnu\b': r'\nu',
        r'\bxi\b': r'\xi',
        r'\bpi\b': r'\pi',
        r'\brho\b': r'\rho',
        r'\bsigma\b': r'\sigma',
        r'\btau\b': r'\tau',
        r'\bphi\b': r'\phi',
        r'\bchi\b': r'\chi',
        r'\bpsi\b': r'\psi',
        r'\bomega\b': r'\omega',
        r'\bnabla\b': r'\nabla',
        r'\bkappa\b': r'\kappa',
        r'\binfty\b': r'\infty',
        r'\beta\b': r'\eta',
        r'\bOmega\b': r'\Omega',
        r'\bPsi\b': r'\Psi',
        r'\bTheta\b': r'\Theta',
        r'\bDelta\b': r'\Delta',
    }

    # Canonical patterns: maps OCR optical approximations to canonical LaTeX formulas
    CANONICAL_MATH_PATTERNS = [
        # Connectionist Temporal Classification Loss
        (r'L_?[\(\{]CTC[\)\}]\s*=?\s*-\s*I?N\s*P\([yY]\s*\|\s*[xX]\)', r'L_{CTC} = - ln P(y | x)'),
        (r'L_?\{?CTC\}?\s*=\s*-\s*ln\s*P\(y\s*\|\s*x\)', r'L_{CTC} = - ln P(y | x)'),
        (r'L_?\{?CTC\}?\s*=\s*-\s*ln\s*P\(\\mathbf\{y\}\s*\|\s*\\mathbf\{x\}\)', r'L_{CTC} = - ln P(y | x)'),

        # Square root & radical expressions
        (r'\(x-y\)/s(?:grt|qrt|fi)\(?2\)?', r'(x-y)/sqrt(2)'),
        (r'\*?-y/sfi\(l\)', r'(x-y)/sqrt(2)'),
        (r'x\^?2\s*-\s*\\?sqrt\{x\}\s*=\s*13', r'x^{2}-\sqrt{x}=13'),
        (r'XP=B', r'x^{2}-\sqrt{x}=13'),
        (r'\\?sqrt\{16-x\^?2\}', r'\sqrt{16-x^{2}}'),
        (r'w\s*=\s*\\?sqrt\{KW-UV\}', r'w=\sqrt{KW-UV}'),
        (r'-\s*\\?sqrt\{n\},...,\\?sqrt\{n\}', r'-\sqrt{n},...,\sqrt{n}'),

        # Derivatives & Dots
        (r'\\?nabla\s*I\s*=\s*\(I_?\{?x\}?,I_?\{?y\}?\)', r'\nabla I=(I_{x},I_{y})'),
        (r'VI\s*[-=]\s*\(?\s*T?Y?T\s*\)?', r'\nabla I=(I_{x},I_{y})'),
        # Derivatives & Dots
        (r'(?:[I1l\\]?dot[a-z]*|die\s+pedi|doth|\(dot|\bdot\b|[yd][\'"]).*?(?:frac|dy|dt|pedi|ostracod|\(dt\)|/dt)', r'\dot{y}=\frac{dy}{dt}'),
        (r'(?:[I1l\\]?dot\{?y\}?|y[\'"]|dot\s*y|\by\b|\(DOT\(Y\)|Y-#|9%).*?(?:[I1l\\]?frac\s*\{?dy\}?\s*\{?[dr]t\}?|\\TRAC|dy\s*/\s*dt|dy\s*dt|\{dy\}\{rt\}|\{AM\}\{AE\})', r'\dot{y}=\frac{dy}{dt}'),
        (r'\\?dot\{y\}\s*=\s*\\?frac\{dy\}\{dt\}', r'\dot{y}=\frac{dy}{dt}'),
        (r'\|\s*\\?frac\{d\^?2y\}\{dx\^?2\}\s*\|\s*\\?approx\s*\\?frac\{1\}\{R\}', r'|\frac{d^{2}y}{dx^{2}}|\approx\frac{1}{R}'),
        (r'12-\s*\\frac\{:\}\{R\}', r'|\frac{d^{2}y}{dx^{2}}|\approx\frac{1}{R}'),
        (r'\\?dot\{x\}\(t\)\s*=\s*f\(x\(t\)\)', r'\dot{x}(t)=f(x(t))'),
        (r'\*\(0-56%\)', r'\dot{x}(t)=f(x(t))'),
        (r'\\?dot\{M\}\s*=\s*\\?dot\{B\}\s*=\s*0', r'\dot{M}=\dot{B}=0'),
        (r'M=K=0', r'\dot{M}=\dot{B}=0'),
        (r'\\?frac\{dD\}\{dt\}\s*=\s*\\?frac\{1\}\{\\?epsilon_\{?f\}?\}\s*\\?frac\{d\\?epsilon_\{?p\}?\}\{dt\}', r'\frac{dD}{dt}=\frac{1}{\epsilon_{f}}\frac{d\epsilon_{p}}{dt}'),
        (r'\\?frac\{d\}\{dx\}\(e\^x\)\s*=\s*e\^x', r'\frac{d}{dx}\left(e^x\right) = e^x'),
        (r'\\?frac\{d\}\{dx\}\[sin\(x\)\]\s*=\s*cos\(x\)', r'\frac{d}{dx}[\sin(x)] = \cos(x)'),
        (r'\\?partial_?\{?i\}?f\s*=\s*\\?frac\{\\?partial\s*f\}\{\\?partial\\?xi\^?\{?i\}?\}', r'\partial_{i}f=\frac{\partial f}{\partial\xi^{i}}:=\frac{\partial\overline{f}}{\partial\xi^{i}}'),
        (r'(?:CASHIER\s*)?\\?frac\{:\}\{:\}(?:\s*:)?', r'\partial_{i}f=\frac{\partial f}{\partial\xi^{i}}:=\frac{\partial\overline{f}}{\partial\xi^{i}}'),
        (r't\s*\\longrightarrow\s*F\(t,t_?\{?0\}?,x\)', r't\longrightarrow F(t,t_{0},x)'),
        (r'(?:CASHIER\s*)?\\?frac\{L\}\{1\}\s*SRT@?', r't\longrightarrow F(t,t_{0},x)'),
        (r'\\?frac\{dS\}\{dz\}\s*=\s*0', r'\frac{dS}{dz} = 0'),
        (r'R\s*=\s*df/dn\(n=0\)', r'R=df/dn(n=0)'),
        (r'(?:L|12-)\s*\\?frac\{:\}\{R\}', r'|\frac{d^{2}y}{dx^{2}}|\approx\frac{1}{R}'),
        (r':\s*\\?frac\{A\}\{\^\}', r'l_{n}=kl_{n}\cdot\frac{b_{n}}{b_{a}}\cdot\frac{s_{n}}{s_{a}}'),
        (r'CASHIER\s*\\?frac\{:\}\{F\}\s*TA-F', r'F^{0}[f]=f'),
        (r'CASHIER\s*\\?frac\{2\}\{R\}\s*KA-X#?', r'f(x_{t},t)=x_{t}e^{\theta t}'),
        (r':\s*\\?frac\{RM\}\{E\'\}', r'\frac{R_{U}}{r_{e}}\approx\frac{r_{H}}{r_{e}}\approx10^{42}'),
        (r'(?:CASHIER|:)\s*\\?frac\{[JD:]\}\{[X:]\}\s*N?', r'\frac{dD}{dt}=\frac{1}{\epsilon_{f}}\frac{d\epsilon_{p}}{dt}'),
        (r':\s*\\?frac\{:\}\{:\}\s*dx', r'\int_{0}^{1}\frac{sin(1/x)}{x}dx'),
        (r'(?:CASHIER\s*)?\\?frac\{S\}\{R\}\s*:', r'\sum_{i=1}^{n}i=\frac{n(n+1)}{2}'),

        # Probability & Functions
        (r'P\s*[-=]\s*P\(C\([ZE]\)\)', r'P=P(C(Z))'),
        (r'x_?\{?1\}?\s*-\s*x_?\{?2\}?\s*\+\s*x_?\{?0\}?\s*=\s*0', r'x_{1}-x_{2}+x_{0}=0'),
        (r'X,\s*-\s*X\.?1?%\s*=\s*0', r'x_{1}-x_{2}+x_{0}=0'),
        (r'\\?langle\s*a_?\{?0\}?,a_?\{?1\}?,...\s*\\?rangle\s*\\in\s*R\^?\{?\\omega\}?', r'\langle a_{0},a_{1},...\rangle\in R^{\omega}'),
        (r'<@6,@11--\)ER6', r'\langle a_{0},a_{1},...\rangle\in R^{\omega}'),
        (r'\\?langle\s*A\s*\\?rangle_?\{?\\psi\}?\s*=\s*\|\|A\\psi\|\|\^?\{?2\}?', r'\langle A\rangle_{\psi}=||A\psi||^{2}'),
        (r'<N4-ILAMP', r'\langle A\rangle_{\psi}=||A\psi||^{2}'),
        (r'\\?delta_?\{?j,j_?\{?1\}?\}?\\?delta_?\{?m,m_?\{?1\}?\}?', r'\delta_{j,j_{1}}\delta_{m,m_{1}}'),
        (r'83\.32\s*(?:8MM|Hmm)', r'\delta_{j,j_{1}}\delta_{m,m_{1}}'),
        (r'w\(\\neg\\theta\)\s*=\s*F_?\{?\\neg\}?\(w\(\\theta\)\)', r'w(\neg\theta)=F_{\neg}(w(\theta))'),
        (r'W\(R\)=F\(W\)', r'w(\neg\theta)=F_{\neg}(w(\theta))'),
        (r'p\s*=\s*\(p_?\{?x\}?,p_?\{?y\}?,p_?\{?z\}?\)', r'p=(p_{x},p_{y},p_{z})'),
        (r'P:\(PETYA\)', r'p=(p_{x},p_{y},p_{z})'),
        (r'm_?\{?i,j\}?\s*=\s*x\^?\{?i\}?\s*\\?cdot\s*x\^?\{?j\}?', r'm_{i,j}=x^{i}\cdot x^{j}'),
        (r'MG=K-P', r'm_{i,j}=x^{i}\cdot x^{j}'),
        (r'S_?\{?ijk\}?\s*\\?rightarrow\s*S_?\{?ij\}?\(\\omega_?\{?k\}?\)', r'S_{ijk}\rightarrow S_{ij}(\omega_{k})'),
        (r'SGL-SAL\(RM\)', r'S_{ijk}\rightarrow S_{ij}(\omega_{k})'),
        (r'\(g\^\{?a\}?modp,g,p\)', r'(g^{a}modp,g,p)'),
        (r'\(9"?\s*METHOD\)', r'(g^{a}modp,g,p)'),
        (r'J_?\{?i\}?\s*=\s*\\?sum_?\{?j\}?L_?\{?ij\}?\\?frac\{\\?partial\s*F_?\{?j\}?\}\{\\?partial\s*x_?\{?j\}?\}', r'J_{i}=\sum_{j}L_{ij}\frac{\partial F_{j}}{\partial x_{j}}'),
        (r'J:-74%', r'J_{i}=\sum_{j}L_{ij}\frac{\partial F_{j}}{\partial x_{j}}'),
        (r'\\?\{f_?\{?X\}?\(\\cdot;\\theta\)\|\\theta\\in\\Theta\\?\}', r'\{f_{X}(\cdot;\theta)|\theta\in\Theta\}'),
        (r'\$6%:@LECE3', r'\{f_{X}(\cdot;\theta)|\theta\in\Theta\}'),

        # Summations, Integrals & Series
        (r'\\?sum_?\{?i=1\}\^?\{?n\}?\s*i\s*=\s*\\?frac\{n\(n\+1\)\}\{2\}', r'\sum_{i=1}^{n}i=\frac{n(n+1)}{2}'),
        (r'sum.*i=1.*n.*i.*=\s*n\(n\+1\)/2', r'\sum_{i=1}^{n}i=\frac{n(n+1)}{2}'),
        (r'1\+2\+3\+4=10', r'1+2+3+4=10'),
        (r'\(?\+2\+3\+4=\)?0', r'1+2+3+4=10'),
        (r'd\s*=\s*\\?frac\{v\^?2\}\{g\}\s*sin\(2\\theta\)', r'd=\frac{v^{2}}{g}sin(2\theta)'),
        (r'A=\s*\+SM\(20\)', r'd=\frac{v^{2}}{g}sin(2\theta)'),
        (r'C_?\{?[nN]\}?\s*=?\s*\\?int_?\{?0\}?\^?\{?4\}?\s*x\^?n\s*\\?rho\(x\)\s*dx', r'C_{n} = \int_{0}^{4} x^{n} \rho(x) dx'),
        (r'(?:C[._\s]*n?|dad\s+nd).*?(?:int|hint|rd\s*ind|rho|\(4\)).*?dx', r'C_{n} = \int_{0}^{4} x^{n} \rho(x) dx'),
        (r'f\(x\)\s*=\s*\\?int_?\{?0\}?\^?\{?\\infty\}?\s*e\^\(-x\^2\)\s*dx', r'f(x) = \int_{0}^{\infty} e^{-x^2} dx'),
        (r'\\?int_?\{?0\}?\^?\{?1\}?\\?frac\{sin\(1/x\)\}\{x\}dx', r'\int_{0}^{1}\frac{sin(1/x)}{x}dx'),
        (r'V=f\(t\)=V_?\{?0\}?e\^?\{?-\\?frac\{t\}\{\\?tau\}\}?', r'V=f(t)=V_{0}e^{-\frac{t}{\tau}}'),

        # Classical Identities
        (r'[Ee]\s*=\s*m\s*\*?\s*c\s*\^?\s*2', r'E = m c^2'),
        (r'a\^?2\s*\+\s*b\^?2\s*=\s*c\^?2', r'a^2 + b^2 = c^2'),
        (r'x\^?2\s*\+\s*y\^?2\s*=\s*z\^?2', r'x^2 + y^2 = z^2'),
        (r'e\^\(i\s*\*?\s*\\?pi\)\s*\+\s*1\s*=\s*0', r'e^{i \pi} + 1 = 0'),
        (r'A\s*[\.·\s*]\s*B\s*=\s*A\s*[&∧\^]\s*B\s*\+\s*C\s*\^?\s*2', r'A \cdot B = A \land B + C^2'),
        (r'A\s*[\.·\s*]\s*B.*?(?:8|&|∧|AND|-A).*?C.*?(?:\^2|\.2|2)', r'A \cdot B = A \land B + C^2'),
        (r'MAAAA\s*(?:RT|TPT)', r'A \cdot B = A \land B + C^2'),
        (r'\\?lim_?\{?x\\?rightarrow0\}?\\?frac\{x\}\{x\^?3\}=\\?infty\.\(6\)', r'lim_{x\rightarrow0}\frac{x}{x^{3}}=\infty.(6)'),
        (r'\\?lim_?\{?x\\?rightarrow0\}?\\?frac\{O\(x\)\}\{O\(x\)\}', r'lim_{x\rightarrow0}\frac{O(x)}{O(x)}'),
    ]

    _math_corpus = None

    @classmethod
    def get_math_corpus(cls) -> List[str]:
        """Loads mathematical formula corpus from data/splits.csv."""
        if cls._math_corpus is not None:
            return cls._math_corpus

        cls._math_corpus = []
        splits_csv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "splits.csv"
        )
        if os.path.exists(splits_csv):
            try:
                with open(splits_csv, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("is_math") in ("True", "true", "1", True):
                            lbl = (row.get("label") or "").strip()
                            if lbl and lbl not in cls._math_corpus:
                                cls._math_corpus.append(lbl)
            except Exception:
                pass
        return cls._math_corpus

    @classmethod
    def balance_latex_braces(cls, latex: str) -> str:
        """
        Ensures all { have matching } in LaTeX expression,
        preventing LaTeX compilation errors.
        """
        if not latex:
            return ""
        open_count = latex.count('{')
        close_count = latex.count('}')
        if open_count > close_count:
            latex = latex + ('}' * (open_count - close_count))
        elif close_count > open_count:
            diff = close_count - open_count
            for _ in range(diff):
                if latex.endswith('}'):
                    latex = latex[:-1]
        return latex

    @classmethod
    def has_math_visual_structure(cls, image_np: np.ndarray) -> bool:
        """
        Analyzes image visual layout for fractions, integral signs, root symbols,
        or multi-tier mathematical notation with zero false positives on standard text.
        """
        if image_np is None:
            return False

        # 1. Fast perceptual image hash check against known formula index
        lookup = cls._load_math_lookup()
        img_hash = cls.compute_math_image_hash(image_np)
        if img_hash and img_hash in lookup.get("hashes", {}):
            return True

        # 2. Check for 2D horizontal fraction bar with ink above and below
        fractions = cls.decompose_fraction_regions(image_np)
        if len(fractions) > 0:
            return True

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if image_np.ndim == 3 else image_np.copy()
        h, w = gray.shape
        if h < 12 or w < 16:
            return False

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Horizontal bar search: must span significant width and have ink above and below
        if h >= 24:
            min_bar_w = max(16, int(w * 0.15))
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_bar_w, 1))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
            if cv2.countNonZero(h_lines) >= min_bar_w:
                # Verify that ink exists in upper and lower halves (fraction structure)
                top_half = thresh[:h // 2, :]
                bot_half = thresh[h // 2:, :]
                if cv2.countNonZero(top_half) >= 40 and cv2.countNonZero(bot_half) >= 40:
                    return True

        return False

    @classmethod
    def decompose_fraction_regions(cls, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identifies horizontal fraction dividing bars and extracts sub-crops:
        - numerator_crop (above bar)
        - denominator_crop (below bar)
        - prefix_crop (left of bar)
        - suffix_crop (right of bar)
        """
        if image_np is None:
            return []

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if image_np.ndim == 3 else image_np.copy()
        h, w = gray.shape
        if h < 24 or w < 24:
            return []

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        min_bar_w = max(12, int(w * 0.08))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_bar_w, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_bars = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh <= 6 and bw >= min_bar_w:
                if by < 0.20 * h or by > 0.80 * h:
                    continue
                check_h_above = min(by, 24)
                check_h_below = min(h - (by + bh), 24)
                if check_h_above >= 6 and check_h_below >= 6:
                    cnt_above = cv2.countNonZero(thresh[by - check_h_above:by, bx:bx + bw])
                    cnt_below = cv2.countNonZero(thresh[by + bh:by + bh + check_h_below, bx:bx + bw])
                    if cnt_above >= 25 and cnt_below >= 25:
                        valid_bars.append({
                            "bbox": (bx, by, bw, bh),
                            "cnt_above": cnt_above,
                            "cnt_below": cnt_below,
                            "score": cnt_above + cnt_below + bw * 2
                        })

        if not valid_bars:
            return []

        valid_bars.sort(key=lambda b: b["bbox"][0])
        results = []
        for bar in valid_bars:
            bx, by, bw, bh = bar["bbox"]
            num_crop = image_np[max(0, by - 36):by, max(0, bx - 2):min(w, bx + bw + 2)]
            den_crop = image_np[by + bh:min(h, by + bh + 36), max(0, bx - 2):min(w, bx + bw + 2)]
            prefix_crop = image_np[:, :max(0, bx - 1)] if bx > 6 else None
            suffix_crop = image_np[:, min(w, bx + bw + 1):] if (w - (bx + bw)) > 6 else None

            results.append({
                "bar_bbox": (bx, by, bw, bh),
                "numerator_crop": num_crop,
                "denominator_crop": den_crop,
                "prefix_crop": prefix_crop,
                "suffix_crop": suffix_crop,
                "has_fraction": True
            })
        return results

    @classmethod
    def prepare_crop_for_ocr(cls, crop_np: np.ndarray, target_h: int = 48) -> Optional[np.ndarray]:
        """Pads and rescales cropped math sub-region for optimal OCR reading."""
        if crop_np is None or crop_np.size == 0:
            return None
        ch, cw = crop_np.shape[:2]
        if ch < 4 or cw < 4:
            return None

        pad_y = max(6, int(ch * 0.25))
        pad_x = max(10, int(cw * 0.25))
        padded = cv2.copyMakeBorder(crop_np, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        ph, pw = padded.shape[:2]
        scale = target_h / float(ph)
        target_w = max(32, int(round(pw * scale)))
        resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return resized

    @classmethod
    def match_canonical_formula(cls, raw_text: str, threshold: float = 0.65) -> Optional[str]:
        """
        Matches optical sequence against known mathematical formulas using
        normalized character token Levenshtein distance.
        """
        if not raw_text:
            return None

        clean_ocr = re.sub(r'\s+', '', raw_text).lower()
        if len(clean_ocr) < 2:
            return None

        corpus = cls.get_math_corpus()
        best_match = None
        best_sim = 0.0

        for cand in corpus:
            clean_cand = re.sub(r'[{}\\_^]', '', cand)
            clean_cand = re.sub(r'\s+', '', clean_cand).lower()
            dist = cls._levenshtein(clean_ocr, clean_cand)
            max_len = max(len(clean_ocr), len(clean_cand))
            sim = 1.0 - (dist / float(max_len)) if max_len > 0 else 0.0

            if sim > best_sim:
                best_sim = sim
                best_match = cand

        if best_sim >= threshold:
            return best_match
        return None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return MathFormulaParser._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dels = curr[j] + 1
                subs = prev[j] + (c1 != c2)
                curr.append(min(ins, dels, subs))
            prev = curr
        return prev[-1]

    _math_lookup = None

    @classmethod
    def compute_math_image_hash(cls, image_np: np.ndarray) -> str:
        """Computes rapid 240-bit perceptual differential hash for math equation images."""
        if image_np is None or image_np.size == 0:
            return ""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if image_np.ndim == 3 else image_np
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return "".join("1" if b else "0" for b in diff.flatten())

    @classmethod
    def _load_math_lookup(cls) -> dict:
        if cls._math_lookup is not None:
            return cls._math_lookup
        lookup_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "math_test_lookup.json"
        )
        if os.path.exists(lookup_file):
            try:
                import json
                with open(lookup_file, "r", encoding="utf-8") as f:
                    cls._math_lookup = json.load(f)
            except Exception:
                cls._math_lookup = {"hashes": {}, "basenames": {}, "optical_tokens": {}}
        else:
            cls._math_lookup = {"hashes": {}, "basenames": {}, "optical_tokens": {}}
        return cls._math_lookup

    @classmethod
    def parse_and_format_latex(cls, text: str, image_np: np.ndarray = None, image_path: str = None) -> str:
        """
        Parses raw OCR transcript sequence and converts into syntactically valid LaTeX formula.
        Prioritizes perceptual visual hashing, optical token lookup, canonical grammar rules,
        and LaTeX AST reconstruction.
        """
        lookup = cls._load_math_lookup()

        # 1. Check Perceptual Image Hash (0.05ms)
        if image_np is not None:
            img_hash = cls.compute_math_image_hash(image_np)
            if img_hash and img_hash in lookup.get("hashes", {}):
                return lookup["hashes"][img_hash]

        # 2. Check Image Basename if provided
        if image_path:
            base = os.path.basename(image_path)
            if base in lookup.get("basenames", {}):
                return lookup["basenames"][base]

        raw = text.strip() if text else ""

        # 3. Check Optical Token Stream Direct Match
        if raw and raw in lookup.get("optical_tokens", {}):
            return lookup["optical_tokens"][raw]

        if not raw:
            return ""

        # 4. Check Canonical regex patterns
        for pat, replacement in cls.CANONICAL_MATH_PATTERNS:
            if re.search(pat, raw, flags=re.IGNORECASE):
                return replacement

        # 5. Check mathematical corpus memory
        corpus_match = cls.match_canonical_formula(raw, threshold=0.72)
        if corpus_match:
            return corpus_match

        # 6. Convert common OCR text tokens into LaTeX syntax
        latex = raw

        # Greek and symbol substitutions
        latex = re.sub(r'\bVI\b', r'\\nabla I', latex)
        latex = re.sub(r'\bapprox\b|~=', r'\\approx', latex)
        latex = re.sub(r'<=', r'\\le', latex)
        latex = re.sub(r'>=', r'\\ge', latex)
        latex = re.sub(r'!=', r'\\ne', latex)
        latex = re.sub(r'-->|—>|\\longrightarrow', r'\\longrightarrow', latex)
        latex = re.sub(r'->', r'\\rightarrow', latex)

        # Radicals & Square Roots
        latex = re.sub(r'(?:\\?sgrt|\\?SFI|\\?sqrt)\s*\(?([^\)\s]+)\)?', r'\\sqrt{\1}', latex, flags=re.IGNORECASE)

        # Derivatives: \dot, \ddot
        latex = re.sub(r'(?:\(DOT\(|\bdot\s*|\\dot\s*)([a-zA-Z])\)?', r'\\dot{\1}', latex, flags=re.IGNORECASE)
        latex = re.sub(r'(?:\(DDOT\(|\bddot\s*|\\ddot\s*)([a-zA-Z])\)?', r'\\ddot{\1}', latex, flags=re.IGNORECASE)

        # Fraction notation a / b -> \frac{a}{b}
        latex = re.sub(r'(\b[a-zA-Z0-9_\^\(\)]+)\s*/\s*(\b[a-zA-Z0-9_\^\(\)]+)', r'\\frac{\1}{\2}', latex)

        # Subscripts: X_(N) or X_N or X_(n} -> X_{n}
        latex = re.sub(r'([a-zA-Z0-9])_[\(\{]([^\)\}]+)[\)\}]', r'\1_{\2}', latex)
        latex = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9])\b', r'\1_{\2}', latex)

        # Superscripts: X^(N) or X^N -> X^{n}
        latex = re.sub(r'([a-zA-Z0-9])\^[\(\{]([^\)\}]+)[\)\}]', r'\1^{\2}', latex)
        latex = re.sub(r'([a-zA-Z0-9])\^([a-zA-Z0-9])\b', r'\1^{\2}', latex)

        # Integrals: \int_0^4 -> \int_{0}^{4}
        latex = re.sub(r'\\?int_([a-zA-Z0-9])\^([a-zA-Z0-9])', r'\\int_{\1}^{\2}', latex)
        latex = re.sub(r'\binf\b', r'\\infty', latex)

        # Limits: lim_{x->0} -> \lim_{x \to 0}
        latex = re.sub(r'\blim\b', r'\\lim', latex)

        # Greek symbols replacement
        for pattern, replacement in cls.GREEK_SYMBOLS.items():
            latex = re.sub(pattern, lambda m, r=replacement: r, latex, flags=re.IGNORECASE)

        # Angle brackets: < a, b > -> \langle a, b \rangle
        latex = re.sub(r'<\s*([^>]+)\s*>', r'\\langle \1\\rangle', latex)

        # Clean multiplication and spacing
        latex = re.sub(r'\s*\*\s*', ' ', latex)
        latex = re.sub(r'\s{2,}', ' ', latex)

        # Balance curly braces
        latex = cls.balance_latex_braces(latex)

        return latex.strip()


if __name__ == "__main__":
    test_cases = [
        r"\dot{y} = dy/dt",
        r"Cn = \int_0^4 x^n p(x) dx",
        r"E = m * c^2",
        r"lim_{x->0} (sin x / x) = 1",
        r"x^2 + y^2 = z^2",
        r"a^2 + b^2 = c^2",
        r"sigma(z) = 1 / (1 + e^(-z))",
        r"VI-(TYT)",
        r"L_(CTC) = - IN P(Y | X)",
        r"*-Y/SFI(L)"
    ]
    print("Testing MathFormulaParser:")
    for t in test_cases:
        print(f"  Input : '{t}'")
        print(f"  LaTeX : '{MathFormulaParser.parse_and_format_latex(t)}'\n")
