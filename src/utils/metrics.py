import time
import torch

class OCRMetrics:
    """
    Utility class to calculate Character Error Rate (CER), Word Error Rate (WER),
    Inference Latency (ms), and Model Parameter Efficiency.
    """
    @staticmethod
    def levenshtein_distance(seq1: str, seq2: str) -> int:
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = [[0] * size_y for _ in range(size_x)]

        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x][y] = matrix[x - 1][y - 1]
                else:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x][y - 1] + 1,
                        matrix[x - 1][y - 1] + 1
                    )
        return matrix[size_x - 1][size_y - 1]

    @classmethod
    def calculate_cer(cls, reference: str, hypothesis: str) -> float:
        if not reference:
            return 0.0 if not hypothesis else 1.0
        distance = cls.levenshtein_distance(reference, hypothesis)
        return round(distance / len(reference), 4)

    @classmethod
    def calculate_wer(cls, reference: str, hypothesis: str) -> float:
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        size_x = len(ref_words) + 1
        size_y = len(hyp_words) + 1
        matrix = [[0] * size_y for _ in range(size_x)]

        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if ref_words[x - 1] == hyp_words[y - 1]:
                    matrix[x][y] = matrix[x - 1][y - 1]
                else:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x][y - 1] + 1,
                        matrix[x - 1][y - 1] + 1
                    )
        return round(matrix[size_x - 1][size_y - 1] / len(ref_words), 4)

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def measure_latency(func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        return result, round(latency_ms, 2)
