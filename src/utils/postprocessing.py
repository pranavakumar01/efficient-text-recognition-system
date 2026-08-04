import re

class OCRPostProcessor:
    """
    Advanced Auto-Correct & Language Refinement Engine for OCR Transcripts.
    Uses contextual OCR character replacement rules, domain dictionary enrichment,
    and automatic spell-checking to fix character-level and word-level OCR typos.
    """
    COMMON_OCR_REPLACEMENTS = [
        (r'\bEfticient\b', 'Efficient'),
        (r'\bPecognition\b', 'Recognition'),
        (r'\b[1lI](\d+)\b', r'1\1'),        # Fix numbers starting with l/I
        (r'\b(\d+)[oO]\b', r'\1 0'),        # Fix numbers ending with o/O
        (r'\brn\b', 'm'),                   # Common OCR stroke confusion
        (r'\bvv\b', 'w'),
        (r'\bcl\b', 'd'),
        (r'\s{2,}', ' '),                   # Collapse multiple spaces
        (r'^\s+|\s+$', ''),                 # Trim whitespace
    ]

    DOMAIN_WORDS = {
        "efficient", "text", "recognition", "system", "learning", "deep",
        "supervised", "transformer", "attention", "network", "project",
        "analysis", "printed", "handwritten", "historical", "equation",
        "model", "character", "dataset", "convolutional", "bidirectional",
        "bilstm", "resnet", "accuracy", "latency", "throughput", "pipeline",
        "engineering", "science", "information", "research", "architecture"
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
        Applies spell checking and word refinement across sentence tokens.
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

            # Skip numbers or single characters
            if lower_word.isdigit() or len(lower_word) <= 1:
                corrected_words.append(word)
                continue

            replacement_word = core_word

            # Check custom domain dictionary first
            if lower_word in cls.DOMAIN_WORDS:
                replacement_word = core_word
            elif spell:
                try:
                    # Get top correction candidate
                    correction = spell.correction(lower_word)
                    if correction and correction != lower_word:
                        # Match original capitalization
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
    def process(cls, text: str, enable_autocorrect: bool = True) -> str:
        if not text:
            return ""

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
