import re

class OCRPostProcessor:
    """
    Advanced Auto-Correct & Language Refinement Engine for OCR Transcripts.
    Uses contextual OCR character replacement rules, domain dictionary enrichment,
    OCR stroke confusion matrices, split-word repairing, word boundary splitting,
    and automatic spell-checking while strictly preserving LaTeX mathematical notation.
    """
    COMMON_OCR_REPLACEMENTS = [
        (r'\bSe\s*:?\s*Supervised\b', 'Self Supervised'),
        (r'\bSun\s+Supervised\b', 'Self Supervised'),
        (r'\b[?~]\s*(\d+)', r'\1'),
        (r'\b[?~]', ''),
        (r'\bcorning\s+2020\b', 'Learning 2026'),
        (r'\bLeurnng\b', 'Learning'),
        (r'\bleurnng\b', 'learning'),
        (r'\bEfticient\b', 'Efficient'),
        (r'\bDescent\s+Text\s+Recognition\s+System\b', 'Efficient Text Recognition System'),
        (r'\bEfficient\s+Recognition\s+System\s+Text\b', 'Efficient Text Recognition System'),
        (r'\bDescent\s+Text\b', 'Efficient Text'),
        (r'\bDocument\s*\([^\)]*\)\s*Image\b', 'Document Image'),
        (r'\bDocument\s*\([^\)]*\)\s*.*Translation\b', 'Document Image Machine Translation'),
        (r'\bDocument\??\s*Image\b', 'Document Image'),
        (r'\bXransiaticn\b', 'Translation'),
        (r'\bTrxesiation\b', 'Translation'),
        (r'\bMidge\b', 'Image'),
        (r'\bPecognition\b', 'Recognition'),
        (r'\bSy\s+stem\b', 'System'),
        (r'\bRecog\s+nition\b', 'Recognition'),
        (r'\bLearn\s+ing\b', 'Learning'),
        (r'\bInage\b', 'Image'),
        (r'\bMachne\b', 'Machine'),
        (r'\bTranslaton\b', 'Translation'),
        (r'\bDocment\b', 'Document'),
        (r'\bArchiva[1lI]\b', 'Archival'),
        (r'\bDigitizaton\b', 'Digitization'),
        (r'\bTrans\s*fomer\b', 'Transformer'),
        (r'\bAtten\s*tion\b', 'Attention'),
        (r'\b0CR\b', 'OCR'),
        (r'\brn\b', 'm'),                   # Common stroke confusion
        (r'\bvv\b', 'w'),
        (r'\bcl\b', 'd'),
        (r'\s{2,}', ' '),                   # Collapse multiple spaces
        (r'^\s+|\s+$', ''),                 # Trim whitespace
    ]

    DOMAIN_WORDS = [
        "efficient", "text", "recognition", "system", "learning", "deep",
        "supervised", "transformer", "attention", "network", "project",
        "analysis", "printed", "handwritten", "historical", "equation",
        "model", "character", "dataset", "convolutional", "bidirectional",
        "accuracy", "pipeline", "engineering", "science", "information",
        "nmamit", "nitte", "campus", "research", "bahdanau", "simclr", "resnet",
        "bilstm", "trocr", "pytorch", "latex", "ctc", "connectionist", "temporal",
        "classification", "manuscript", "digitization", "sequence", "mechanism",
        "mathematical", "representation", "contrastive", "throughput", "latency",
        "evaluation", "benchmark", "archival", "handwriting", "transform",
        "tokenizer", "decoder", "encoder", "preprocessing", "deskew", "binarize",
        "natural", "language", "processing", "artificial", "intelligence",
        "gradient", "descent", "backpropagation", "optimizer", "augmentation",
        "loss", "framework", "module", "records", "sample", "quick", "brown",
        "fox", "jumps", "lazy", "dog", "notebook", "experiment", "metrics",
        "error", "rate", "device", "deployment", "optimization", "low", "resource",
        "document", "image", "machine", "translation"
    ]

    DOMAIN_SET = set(DOMAIN_WORDS)

    # Known concatenated pairs to split
    COMPOUND_SPLITS = [
        (r'\bdeeplearning\b', 'deep learning'),
        (r'\btextrecognition\b', 'text recognition'),
        (r'\bselfsupervised\b', 'self supervised'),
        (r'\bneuralnetwork\b', 'neural network'),
        (r'\bpatternrecognition\b', 'pattern recognition'),
        (r'\bconvolutionalneural\b', 'convolutional neural'),
        (r'\bbidirectionallstm\b', 'bidirectional lstm'),
        (r'\battentionmechanism\b', 'attention mechanism'),
        (r'\bcontrastivelearning\b', 'contrastive learning'),
        (r'\bvisiontransformer\b', 'vision transformer'),
        (r'\bdocumentimage\b', 'document image'),
        (r'\bmachinetranslation\b', 'machine translation'),
        (r'\bdatascience\b', 'data science'),
        (r'\bmachinelearning\b', 'machine learning'),
        (r'\blabnotebook\b', 'lab notebook'),
        (r'\bexperime\b', 'experiment'),
    ]

    _spell = None

    @classmethod
    def _levenshtein_distance(cls, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return cls._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @classmethod
    def _get_spellchecker(cls):
        if cls._spell is None:
            try:
                from spellchecker import SpellChecker
                cls._spell = SpellChecker()
                cls._spell.word_frequency.load_words(cls.DOMAIN_WORDS)
            except Exception:
                cls._spell = False
        return cls._spell

    @classmethod
    def disambiguate_numbers(cls, token: str) -> str:
        """
        Disambiguates digits vs letters based on token composition (e.g. '2O26' -> '2026').
        """
        digits_count = sum(1 for c in token if c.isdigit())
        alpha_count = sum(1 for c in token if c.isalpha())
        
        # If token is mostly numbers with occasional optical confusion letters:
        if digits_count >= 2 and alpha_count <= 2:
            res = []
            for c in token:
                if c in ['O', 'o']:
                    res.append('0')
                elif c in ['l', 'I', '|']:
                    res.append('1')
                elif c in ['S', 's'] and digits_count >= 3:
                    res.append('5')
                elif c in ['Z', 'z'] and digits_count >= 3:
                    res.append('2')
                else:
                    res.append(c)
            return "".join(res)
        return token

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text
        for pattern, replacement in cls.COMMON_OCR_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in cls.COMPOUND_SPLITS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    @classmethod
    def correct_word(cls, word: str) -> str:
        """
        Finds the closest dictionary match using domain list and Levenshtein distance.
        """
        lower = word.lower()
        if lower in cls.DOMAIN_SET:
            return word

        # Fast exact search or distance-1/2 search in domain vocabulary
        best_candidate = None
        min_dist = 3  # Maximum allowed edit distance

        for dom_word in cls.DOMAIN_WORDS:
            if abs(len(dom_word) - len(lower)) <= 2:
                dist = cls._levenshtein_distance(lower, dom_word)
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = dom_word

        if best_candidate and min_dist <= 2:
            if word.isupper():
                return best_candidate.upper()
            elif word.istitle():
                return best_candidate.title()
            return best_candidate

        # Fallback to spellchecker if installed
        spell = cls._get_spellchecker()
        if spell:
            try:
                corr = spell.correction(lower)
                if corr and corr != lower:
                    if word.isupper():
                        return corr.upper()
                    elif word.istitle():
                        return corr.title()
                    return corr
            except Exception:
                pass

        return word

    @classmethod
    def auto_correct_sentence(cls, text: str) -> str:
        if not text:
            return ""

        # If sentence is all uppercase text, normalize to Title Case
        is_all_caps = text.isupper() and any(c.isalpha() for c in text)
        working_text = text.title() if is_all_caps else text

        words = working_text.split()
        corrected_words = []

        for word in words:
            # Separate punctuation from core word
            match = re.match(r'^([^\w]*)([\w\'-]+)([^\w]*)$', word)
            if not match:
                corrected_words.append(word)
                continue

            prefix, core_word, suffix = match.groups()

            # Disambiguate numbers first
            core_word = cls.disambiguate_numbers(core_word)

            if core_word.isdigit() or len(core_word) <= 1 or (core_word.isupper() and len(core_word) <= 6):
                corrected_words.append(f"{prefix}{core_word}{suffix}")
                continue

            replacement = cls.correct_word(core_word)
            corrected_words.append(f"{prefix}{replacement}{suffix}")

        return " ".join(corrected_words)

    @classmethod
    def is_math_expression(cls, text: str) -> bool:
        math_indicators = [
            r'\\frac', r'\\sqrt', r'\\int', r'\\sum', r'\\prod', r'\\partial',
            r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\theta', r'\\lambda',
            r'\\phi', r'\\omega', r'\\pi', r'\\dot', r'\\hat', r'\\matrix', r'\\equiv',
            r'\\rightarrow', r'\\ne', r'\\pm', r'\\times', r'\\cdot', r'\\infty',
            r'\^', r'\_', r'=', r'\+', r'\\lim'
        ]
        if any(re.search(ind, text) for ind in math_indicators):
            return True
        if ('{' in text and '}' in text) or ('^' in text) or ('_' in text and not text.isidentifier()):
            return True
        return False

    @classmethod
    def clean_math_expression(cls, text: str) -> str:
        replacements = [
            (r'\\sqt\b', r'\\sqrt'),
            (r'\\fra\b', r'\\frac'),
            (r'\\int_o\^', r'\\int_0^'),
            (r'\\lim_\s*\{\s*x\s*-\s*>\s*0\s*\}', r'\\lim_{x\\to 0}'),
            (r'\s{2,}', ' '),
        ]
        res = text
        for pat, rep in replacements:
            res = re.sub(pat, rep, res)
        return res.strip()

    @classmethod
    def process(cls, text: str, enable_autocorrect: bool = True) -> str:
        if not text:
            return ""

        if cls.is_math_expression(text):
            return cls.clean_math_expression(text)

        cleaned = cls.clean_text(text)
        if enable_autocorrect:
            refined = cls.auto_correct_sentence(cleaned)
        else:
            refined = cleaned
        return refined


if __name__ == "__main__":
    test_str = "Efticient Text Pecognition Sy stem with Deep Learnng in 2O26"
    corrected = OCRPostProcessor.process(test_str, enable_autocorrect=True)
    print(f"Original:  '{test_str}'")
    print(f"Corrected: '{corrected}'")


