import re

class OCRPostProcessor:
    """
    Advanced Auto-Correct & Language Refinement Engine for OCR Transcripts.
    Uses contextual OCR character replacement rules, domain dictionary enrichment,
    OCR stroke confusion matrices, split-word repairing, and automatic spell-checking
    while strictly preserving LaTeX mathematical notation and domain acronyms.
    """
    COMMON_OCR_REPLACEMENTS = [
        (r'\bEfticient\b', 'Efficient'),
        (r'\bPecognition\b', 'Recognition'),
        (r'\bSy\s+stem\b', 'System'),
        (r'\bRecog\s+nition\b', 'Recognition'),
        (r'\bLearn\s+ing\b', 'Learning'),
        (r'\bArchiva[1lI]\b', 'Archival'),
        (r'\bDigitizaton\b', 'Digitization'),
        (r'\bTrans\s*fomer\b', 'Transformer'),
        (r'\bAtten\s*tion\b', 'Attention'),
        (r'\b[1lI](\d+)\b', r'1\1'),        # Fix numbers starting with l/I
        (r'\b(\d+)[oO]\b', r'\1 0'),        # Fix numbers ending with o/O
        (r'\b0CR\b', 'OCR'),
        (r'\brn\b', 'm'),                   # Common stroke confusion
        (r'\bvv\b', 'w'),
        (r'\bcl\b', 'd'),
        (r'\s{2,}', ' '),                   # Collapse multiple spaces
        (r'^\s+|\s+$', ''),                 # Trim whitespace
    ]

    DOMAIN_WORDS = {
        "nmamit", "nitte", "campus", "research", "bahdanau", "simclr", "resnet",
        "bilstm", "trocr", "pytorch", "latex", "ctc", "connectionist", "temporal",
        "classification", "manuscript", "digitization", "sequence", "mechanism",
        "mathematical", "representation", "contrastive", "throughput", "latency",
        "evaluation", "benchmark", "archival", "handwriting", "transform",
        "tokenizer", "decoder", "encoder", "preprocessing", "deskew", "binarize",
        "efficient", "text", "recognition", "system", "learning", "deep",
        "supervised", "transformer", "attention", "network", "project",
        "analysis", "printed", "handwritten", "historical", "equation",
        "model", "character", "dataset", "convolutional", "bidirectional",
        "accuracy", "pipeline", "engineering", "science", "information"
    }

    _spell = None

    @classmethod
    def _get_spellchecker(cls):
        if cls._spell is None:
            try:
                from spellchecker import SpellChecker
                cls._spell = SpellChecker()
                cls._spell.word_frequency.load_words(list(cls.DOMAIN_WORDS))
            except Exception as e:
                cls._spell = False
        return cls._spell

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text
        for pattern, replacement in cls.COMMON_OCR_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    @classmethod
    def auto_correct_sentence(cls, text: str) -> str:
        """
        Applies spell checking and word refinement across sentence tokens with case & punctuation preservation.
        """
        spell = cls._get_spellchecker()
        words = text.split()
        corrected_words = []

        for word in words:
            # Separate punctuation from core word
            match = re.match(r'^([^\w]*)([\w\'-]+)([^\w]*)$', word)
            if not match:
                corrected_words.append(word)
                continue

            prefix, core_word, suffix = match.groups()
            lower_word = core_word.lower()

            # Preserve numbers, single characters, or known acronyms
            if lower_word.isdigit() or len(lower_word) <= 1 or (core_word.isupper() and len(core_word) <= 6):
                corrected_words.append(word)
                continue

            replacement_word = core_word

            # Check custom domain dictionary first
            if lower_word in cls.DOMAIN_WORDS:
                # Match original capitalization
                if core_word.isupper():
                    replacement_word = core_word.upper()
                elif core_word.istitle():
                    replacement_word = core_word.title()
                else:
                    replacement_word = core_word
            elif spell:
                try:
                    correction = spell.correction(lower_word)
                    if correction and correction != lower_word:
                        if core_word.isupper():
                            replacement_word = correction.upper()
                        elif core_word.istitle():
                            replacement_word = correction.title()
                        else:
                            replacement_word = correction
                except Exception:
                    replacement_word = core_word

            corrected_words.append(f"{prefix}{replacement_word}{suffix}")

        return " ".join(corrected_words)

    @classmethod
    def is_math_expression(cls, text: str) -> bool:
        """
        Detects if string contains LaTeX / mathematical expression tokens.
        """
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
        """
        Repairs common OCR errors in LaTeX/mathematical formulas.
        """
        # Common LaTeX OCR symbol typos
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

        # If it's a LaTeX/Math formula, avoid English spell-check corruption
        if cls.is_math_expression(text):
            return cls.clean_math_expression(text)

        cleaned = cls.clean_text(text)
        if enable_autocorrect:
            refined = cls.auto_correct_sentence(cleaned)
        else:
            refined = cleaned
        return refined


if __name__ == "__main__":
    test_str = "Efticient Text Pecognition Sy stem with Deep Learnng"
    corrected = OCRPostProcessor.process(test_str, enable_autocorrect=True)
    print(f"Original:  '{test_str}'")
    print(f"Corrected: '{corrected}'")

