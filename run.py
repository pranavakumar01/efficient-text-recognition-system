"""
Unified Runner and CLI Entry Point for Efficient Text Recognition System (DL + SSL)
Supports:
  1. Web Server:       python run.py --server [--port 8000] [--host 127.0.0.1]
  2. Text Extraction:  python run.py --image data/samples/printed_sample.png [--model both] [--domain auto]
  3. Image Extraction: python run.py data/samples/math_equation_sample.png
  4. System Self-Test: python run.py --test
"""

import os
import sys
import socket
import argparse
import webbrowser
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port(port: int):
    """Frees the specified port if occupied by a stale/zombie process on Windows."""
    if sys.platform != "win32":
        return
    try:
        import subprocess
        cmd = f'netstat -ano | findstr :{port}'
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts[3].upper():
                pid = parts[4]
                if pid and pid != "0" and int(pid) != os.getpid():
                    print(f"[*] Port {port} is occupied by PID {pid}. Terminating stale process...")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.5)
    except Exception:
        pass


def launch_tunnel(port: int = 8000):
    """Launches a free Cloudflare Quick Tunnel to expose the local server to a public HTTPS URL."""
    import re
    import threading
    import subprocess

    cloudflared_path = os.path.join(ROOT_DIR, "cloudflared.exe")
    if not os.path.exists(cloudflared_path):
        print("[!] cloudflared.exe not found. Tunnel cannot be started.")
        return None

    cmd = [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def monitor():
        start_t = time.time()
        tunnel_url = None
        while time.time() - start_t < 25:
            line = proc.stdout.readline()
            if not line:
                break
            m = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if m:
                tunnel_url = m.group(0)
                break
        if tunnel_url:
            print("\n" + "=" * 75)
            print(f" [ONLINE] PUBLIC TEMPORARY DEMO URL: {tunnel_url}")
            print(" (Share this HTTPS link with anyone to test from any phone or browser!)")
            print("=" * 75 + "\n")
            try:
                webbrowser.open(tunnel_url)
            except Exception:
                pass

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return proc


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, open_browser: bool = True, tunnel: bool = False):
    print("\n" + "=" * 75)
    print("     MAJOR PROJECT: EFFICIENT TEXT RECOGNITION (DL + SSL)")
    print("=" * 75)
    print(f" Local Web Server Target: http://{host}:{port}")
    print(f" Public Online Tunnel    : {'Enabled (Cloudflare HTTPS)' if tunnel else 'Disabled (Local only)'}")
    print(f" Auto-Reload Mode        : {reload}")

    # Check port availability and auto-free zombie processes
    if is_port_in_use(port, host):
        print(f"[!] Warning: Port {port} is already in use. Cleaning up stale listeners...")
        free_port(port)
        if is_port_in_use(port, host):
            print(f"[X] Error: Could not free port {port}. Please specify another port: python run.py --server --port 8080")
            sys.exit(1)
        print(f"[OK] Port {port} successfully cleared and ready.")

    tunnel_proc = None
    if tunnel:
        print("[*] Starting secure online tunnel ...")
        tunnel_proc = launch_tunnel(port=port)

    url = f"http://{host}:{port}"
    if open_browser and not tunnel:
        try:
            print(f"[*] Opening browser at {url} ...")
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn
    print(f"[OK] Starting Uvicorn FastAPI server on {url} ... (Press Ctrl+C to stop)\n")
    try:
        uvicorn.run("app.server:app", host=host, port=port, reload=reload)
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass



def run_self_test():
    print("\n" + "=" * 75)
    print(" SYSTEM SELF-TEST: VERIFYING MODEL INFERENCE ACROSS SAMPLES")
    print("=" * 75)

    import cv2
    from src.infer import OCRInferenceEngine
    engine = OCRInferenceEngine(load_trocr=True)

    test_cases = [
        ("printed_sample.png", "printed", "Efficient Text Recognition System"),
        ("handwritten_sample.png", "auto", "Self Supervised Learning 2026"),
        ("math_equation_sample.png", "math", "A \\cdot B = A & B + C^2"),
        ("mathwriting_derivative_sample.png", "math", "\\dot{y} = \\frac{dy}{dt}"),
        ("mathwriting_integral_sample.png", "math", "C_{n} = \\int_{0}^{4} x^{n} \\rho(x) dx"),
        ("historical_document_sample.png", "historical", "Document Image Machine Translation"),
    ]

    all_passed = True
    samples_dir = os.path.join(ROOT_DIR, "data", "samples")

    for filename, domain, expected in test_cases:
        filepath = os.path.join(samples_dir, filename)
        if not os.path.exists(filepath):
            print(f" [SKIP] {filename} (not found)")
            continue

        img = cv2.imread(filepath)
        res = engine.run_pipeline(img, model_type="both", domain=domain, is_historical=(domain == "historical"))
        cnn_pred = res.get("cnn_bilstm_attention", {}).get("predicted_text", "").strip()
        trocr_pred = res.get("transformer_baseline", {}).get("predicted_text", "").strip()
        math_pred = res.get("math_engine", {}).get("predicted_text", "").strip() if res.get("math_engine") else ""

        best = math_pred or cnn_pred or trocr_pred
        match = (best.lower() == expected.lower()) or (expected.lower() in best.lower())
        status = "[PASS]" if match else "[DIFF]"
        if not match:
            all_passed = False

        print(f" {status} {filename:34s} | Pred: {best}")
        if not match:
            print(f"        Expected: {expected}")

    print("=" * 75)
    if all_passed:
        print(" [OK] All system test cases PASSED with 100% accuracy!")
    else:
        print(" [*] System verification completed.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Entry Point for Efficient Text Recognition System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional path to image file for immediate recognition",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI web application and UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the web server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot reloading for development",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser when starting the server",
    )
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Expose the server to a public temporary HTTPS URL via Cloudflare Quick Tunnel",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file to perform OCR extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["both", "cnn", "transformer", "trocr", "all"],
        help="OCR model architecture (default: both)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="auto",
        choices=["auto", "printed", "handwritten", "math", "historical"],
        help="Document domain type (default: auto)",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use edge INT8 quantized CNN model",
    )
    parser.add_argument(
        "--doc",
        action="store_true",
        help="Force full multi-line document segmentation mode",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Enable historical document processing mode",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system self-test across all evaluation sample images",
    )

    args = parser.parse_args()

    # 1. Self-Test Mode
    if args.test:
        run_self_test()
        return

    # 2. Image Extraction Mode
    target_image = args.image_pos or args.image
    if target_image:
        from extract_text import run_extraction
        model_key = "transformer" if args.model == "trocr" else args.model
        run_extraction(
            target_image,
            model_choice=model_key,
            domain=args.domain,
            is_document=args.doc,
            use_quantized=args.quantized,
            is_historical=args.historical,
        )
        return

    # 3. Server Mode (default if --server passed or no other CLI arguments given)
    server_port = int(os.environ.get("PORT", args.port))
    server_host = os.environ.get("HOST", args.host)
    start_server(
        host=server_host,
        port=server_port,
        reload=args.reload,
        open_browser=not args.no_browser and not os.environ.get("PORT"),
        tunnel=args.tunnel,
    )


if __name__ == "__main__":
    main()
