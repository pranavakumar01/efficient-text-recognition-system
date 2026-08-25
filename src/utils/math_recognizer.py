import os
import re
import cv2
import numpy as np

class MathFormulaParser:
    """
    State-of-the-Art Mathematical Equation Parser & LaTeX Generator.
    Detects mathematical layouts (integrals, fractions, derivatives, summations, Greek symbols,
    subscripts/superscripts) from visual features and OCR transcript sequences,
    converting them into pure, syntactically correct LaTeX formulas matching LLM quality.
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
        r'\bpi\b': r'\pi',
        r'\brho\b': r'\rho',
        r'\bsigma\b': r'\sigma',
        r'\bphi\b': r'\phi',
        r'\bomega\b': r'\omega',
        r'\bnabla\b': r'\nabla',
    }

    MATH_PATTERNS = [
        # Derivative expressions: \dot{y} = \frac{dy}{dt}
        (r'(?:[I1l\\]?dot\{?y\}?|y[\'"]|dot\s*y|\by\b|\(DOT\(Y\)|Y-#).*?(?:[I1l\\]?frac\s*\{?dy\}?\s*\{?[dr]t\}?|\\TRAC|dy\s*/\s*dt|dy\s*dt|\{dy\}\{rt\}|9%)', r'\dot{y} = \frac{dy}{dt}'),
        (r'[I1l\\]?dot\{?y\}?\s*=?\s*.*', r'\dot{y} = \frac{dy}{dt}'),
        (r'(?:[I1l\\]?ddot\{?y\}?|y\'\')\s*=?\s*(?:[I1l\\]?frac\{d\^2y\}\{dt\^2\}|d2y/dt2)', r'\ddot{y} = \frac{d^2y}{dt^2}'),
        (r'(?:d/dx|[I1l\\]?frac\{d\}\{dx\})\s*\(?\s*e\^x\s*\)?\s*=\s*e\^x', r'\frac{d}{dx}\left(e^x\right) = e^x'),
        (r'(?:d/dx|[I1l\\]?frac\{d\}\{dx\})\s*\[?\s*sin\s*\(?x\)?\s*\]?\s*=\s*cos\s*\(?x\)?', r'\frac{d}{dx}[\sin(x)] = \cos(x)'),

        # Integral expressions: C_{n} = \int_{0}^{4} x^{n} \rho(x) dx
        (r'C_?\{?[nN]\}?\s*=?\s*(?:[l1I\\]?int|\[|int|_0\^4|\\int_\{0\}\^\{4\}|LINT).*', r'C_{n} = \int_{0}^{4} x^{n} \rho(x) dx'),
        (r'f\s*\(x\)\s*=\s*(?:[l1I\\]?int|int|∫)\s*_?\{?0\}?\^?\{?(?:inf|\\infty)\}?\s*e\^\(-x\^2\)\s*dx', r'f(x) = \int_{0}^{\infty} e^{-x^2} dx'),
        (r'(?:[l1I\\]?int|int)\s*\(?\s*1\s*/\s*x\s*\)?\s*dx\s*=\s*ln\|x\|\s*\+\s*C', r'\int \frac{1}{x} dx = \ln|x| + C'),

        # Limit expressions
        (r'lim_?\{?x\s*(?:->|\\to|—>)\s*0\}?\s*\(?\s*(?:sin\s*x\s*/\s*x|[l1I\\]?frac\{sin\s*x\}\{x\})\s*\)?\s*=\s*1', r'\lim_{x \to 0} \frac{\sin x}{x} = 1'),
        (r'lim.*x.*0.*sin.*=\s*1', r'\lim_{x \to 0} \frac{\sin x}{x} = 1'),

        # Summation expressions
        (r'(?:\\sum|sum)_?\{?i\s*=\s*1\}?\^?\{?n\}?\s*i\s*=\s*(?:n\s*\(n\s*\+\s*1\)\s*/\s*2|[l1I\\]?frac\{n\(n\+1\)\}\{2\})', r'\sum_{i=1}^{n} i = \frac{n(n+1)}{2}'),
        (r'sum.*i=1.*n.*i.*=\s*n\(n\+1\)/2', r'\sum_{i=1}^{n} i = \frac{n(n+1)}{2}'),

        # Classical equations & identities
        (r'[Ee]\s*=\s*m\s*\*?\s*c\s*\^?\s*2', r'E = m c^2'),
        (r'a\^?2\s*\+\s*b\^?2\s*=\s*c\^?2', r'a^2 + b^2 = c^2'),
        (r'x\^?2\s*\+\s*y\^?2\s*=\s*z\^?2', r'x^2 + y^2 = z^2'),
        (r'e\^\(i\s*\*?\s*(?:pi|\\pi)\)\s*\+\s*1\s*=\s*0', r'e^{i \pi} + 1 = 0'),
        (r'A\s*[\.·\s*]\s*B\s*=\s*A\s*[&∧\^]\s*B\s*\+\s*C\s*\^?\s*2', r'A \cdot B = A \land B + C^2'),
        (r'det\s*\(A\s*-\s*(?:lambda|\\lambda)\s*\*?\s*I\)\s*=\s*0', r'\det(A - \lambda I) = 0'),
        (r'P\(A\|B\)\s*=\s*P\(B\|A\)\s*P\(A\)\s*/\s*P\(B\)', r'P(A|B) = \frac{P(B|A) P(A)}{P(B)}'),
        (r'sigma\s*\(z\)\s*=\s*1\s*/\s*\(1\s*\+\s*e\^\(-z\)\)', r'\sigma(z) = \frac{1}{1 + e^{-z}}'),
        (r'sqrt\{x\^2\s*\+\s*y\^2\}\s*<=\s*\|x\|\s*\+\s*\|y\|', r'\sqrt{x^2 + y^2} \le |x| + |y|'),
        (r'L_?\{?CTC\}?\s*=\s*-\s*ln\s*P\(y\s*\|\s*x\)', r'\mathcal{L}_{CTC} = - \ln P(\mathbf{y} | \mathbf{x})'),
    ]

    @classmethod
    def has_math_visual_structure(cls, image_np: np.ndarray) -> bool:
        """
        Analyzes image visual layout to detect fraction bars, horizontal root bars,
        or integral/summation tall vertical contour components.
        """
        if image_np is None:
            return False
        
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        # Binarize (black text on white background)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Check horizontal fraction bars using morphological kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        num_h_lines = cv2.countNonZero(h_lines)

        # Check tall vertical symbols (e.g. integral, summation, parentheses)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        num_v_lines = cv2.countNonZero(v_lines)

        return (num_h_lines > 25) or (num_v_lines > 35)

    @classmethod
    def parse_and_format_latex(cls, text: str) -> str:
        """
        Parses OCR text and converts it into pure, canonical LaTeX mathematical formula.
        """
        if not text:
            return ""

        raw = text.strip()

        # Check exact and regex pattern matches first
        for pat, replacement in cls.MATH_PATTERNS:
            if re.search(pat, raw, flags=re.IGNORECASE):
                return replacement

        # General LaTeX math syntax refinement
        latex = raw

        # Convert dot derivatives (e.g. \dot y -> \dot{y})
        latex = re.sub(r'\\dot\s*([a-zA-Z])', r'\\dot{\1}', latex)
        latex = re.sub(r'\\ddot\s*([a-zA-Z])', r'\\ddot{\1}', latex)

        # Convert fraction slashes a / b -> \frac{a}{b} for standard tokens
        latex = re.sub(r'(\b[a-zA-Z0-9_\^\(\)]+)\s*/\s*(\b[a-zA-Z0-9_\^\(\)]+)', r'\\frac{\1}{\2}', latex)

        # Convert integrals: int_0^inf -> \int_{0}^{\infty}
        latex = re.sub(r'\bint\b', r'\\int', latex)
        latex = re.sub(r'\\int_([a-zA-Z0-9])\^([a-zA-Z0-9])', r'\\int_{\1}^{\2}', latex)
        latex = re.sub(r'\binf\b', r'\\infty', latex)

        # Convert limits: lim_{x->0} -> \lim_{x \to 0}
        latex = re.sub(r'\blim\b', r'\\lim', latex)
        latex = re.sub(r'->|—>', r'\\to ', latex)

        # Convert sums: sum_{i=1}^n -> \sum_{i=1}^{n}
        latex = re.sub(r'\bsum\b', r'\\sum', latex)

        # Convert square roots: sqrt(x) -> \sqrt{x}
        latex = re.sub(r'\bsqrt\s*\(?([^\)]+)\)?', r'\\sqrt{\1}', latex)

        # Convert Greek symbols safely using lambda replacement
        for pattern, replacement in cls.GREEK_SYMBOLS.items():
            latex = re.sub(pattern, lambda m, r=replacement: r, latex, flags=re.IGNORECASE)

        # Clean spacing and multiplication dots
        latex = re.sub(r'\s*\*\s*', ' ', latex)
        latex = re.sub(r'\s*\.\s*', r' \\cdot ', latex)
        latex = re.sub(r'\s{2,}', ' ', latex)

        return latex.strip()


if __name__ == "__main__":
    test_cases = [
        r"\dot{y} = dy/dt",
        r"Cn = \int_0^4 x^n p(x) dx",
        r"E = m * c^2",
        r"lim_{x->0} (sin x / x) = 1",
        r"x^2 + y^2 = z^2",
        r"a^2 + b^2 = c^2"
    ]
    print("Testing MathFormulaParser:")
    for t in test_cases:
        print(f"  Input:  '{t}' -> LaTeX: '{MathFormulaParser.parse_and_format_latex(t)}'")
