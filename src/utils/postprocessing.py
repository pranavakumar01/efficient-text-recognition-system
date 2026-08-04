import re

class OCRPostProcessor:
    """
    Language & Character Post-Processing Module for OCR Predictions.
    Refines raw decoder outputs using dictionary lookups, whitespace normalization,
    and rule-based typo correction.
    """
    COMMON_OCR_REPLACEMENTS = [
        (r'\b[1lI](\d+)\b', r'1\1'),        # Fix numbers starting with l/I
        (r'\b(\d+)[oO]\b', r'\1 0'),        # Fix numbers ending with o/O
        (r'\s{2,}', ' '),                   # Collapse multiple spaces
        (r'^\s+|\s+$', ''),                 # Trim whitespace
    ]

    COMMON_WORDS = {
        "sample", "document", "recognition", "learning", "deep",
        "supervised", "transformer", "attention", "network", "project",
        "analysis", "printed", "handwritten", "historical", "equation",
        "text", "system", "accuracy", "model", "character", "dataset"
    }

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text
        for pattern, replacement in cls.COMMON_OCR_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    @classmethod
    def refine_words(cls, text: str) -> str:
        """
        Refines individual tokens against known domain vocabulary.
        """
        words = text.split()
        refined_words = []
        for word in words:
            clean_w = word.strip(".,;:!?()[]{}")
            lower_w = clean_w.lower()
            
            # Simple edit distance check if word is close to common dictionary word
            matched = False
            if len(lower_w) > 3 and lower_w not in cls.COMMON_WORDS:
                for dict_word in cls.COMMON_WORDS:
                    if cls._levenshtein_distance(lower_w, dict_word) == 1:
                        # Match case
                        if clean_w.isupper():
                            rep = dict_word.upper()
                        elif clean_w.istitle():
                            rep = dict_word.title()
                        else:
                            rep = dict_word
                        word = word.replace(clean_w, rep)
                        matched = True
                        break
            refined_words.append(word)

        return " ".join(refined_words)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return OCRPostProcessor._levenshtein_distance(s2, s1)
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
    def process(cls, text: str) -> str:
        cleaned = cls.clean_text(text)
        refined = cls.refine_words(cleaned)
        return refined


if __name__ == "__main__":
    test_str = "  Recognzed  Sample   Tekxt  "
    processed = OCRPostProcessor.process(test_str)
    print(f"Original: '{test_str}' -> Refined: '{processed}'")
