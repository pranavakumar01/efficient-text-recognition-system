import os
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.infer import OCRInferenceEngine
from src.document_ocr import DocumentOCREngine
from data.generate_samples import generate_sample_images

app = FastAPI(
    title="Efficient Text Recognition (DL + SSL)",
    description="Major Project Phase-I Web Demo API & Evaluation System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = OCRInferenceEngine()
doc_engine = DocumentOCREngine(inference_engine=engine)
samples_dir = "d:\\Major Project\\data\\samples"
generate_sample_images(samples_dir)

def encode_img_to_b64(img_np: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/api/samples")
def list_samples():
    sample_files = []
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.endswith(".png"):
                sample_files.append({
                    "name": f.replace("_", " ").replace(".png", "").title(),
                    "filename": f
                })
    return JSONResponse({"samples": sample_files})

@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    filepath = os.path.join(samples_dir, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)
    img_np = cv2.imread(filepath)
    b64 = encode_img_to_b64(img_np)
    return JSONResponse({"image_b64": f"data:image/png;base64,{b64}"})

@app.get("/api/edge_metrics")
def get_edge_metrics():
    summary_path = os.path.join("d:\\Major Project", "docs", "edge_benchmark_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse({"status": "success", "edge_metrics": data})
    return JSONResponse({"status": "pending", "message": "Edge benchmark not yet computed"})

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("both"),
    enable_autocorrect: bool = Form(True),
    ground_truth: str = Form(None),
    domain: str = Form("auto"),
    is_historical: bool = Form(False),
    use_quantized: bool = Form(False)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        global engine
        if use_quantized != getattr(engine, "use_quantized", False):
            engine = OCRInferenceEngine(use_quantized=use_quantized)

        results = engine.run_pipeline(
            image_np,
            model_type=model_type,
            ground_truth=ground_truth,
            enable_autocorrect=enable_autocorrect,
            domain=domain,
            is_historical=is_historical
        )

        b64_preprocessing = {}
        if results.get("preprocessing"):
            for stage_name, stage_img in results["preprocessing"].items():
                if not isinstance(stage_img, np.ndarray):
                    continue
                if len(stage_img.shape) == 2:
                    stage_img = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
                b64_preprocessing[stage_name] = f"data:image/png;base64,{encode_img_to_b64(stage_img)}"

        return JSONResponse({
            "status": "success",
            "active_model": results.get("active_model", model_type),
            "autocorrect_enabled": enable_autocorrect,
            "is_quantized": use_quantized,
            "preprocessing_stages": b64_preprocessing,
            "primary_model": results.get("cnn_bilstm_attention"),
            "baseline_model": results.get("transformer_baseline"),
            "math_engine": results.get("math_engine")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/document_ocr")
async def process_document(
    file: UploadFile = File(...),
    model_type: str = Form("both"),
    enable_autocorrect: bool = Form(True),
    domain: str = Form("auto"),
    is_historical: bool = Form(False),
    use_quantized: bool = Form(False)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        global doc_engine
        out = doc_engine.process_document(
            image_np, model_type=model_type, enable_autocorrect=enable_autocorrect,
            domain=domain, is_historical=is_historical, use_quantized=use_quantized
        )
        return JSONResponse({
            "status": "success",
            "model_type": out["model_type"],
            "autocorrect_enabled": enable_autocorrect,
            "is_quantized": use_quantized,
            "total_lines_detected": out["total_lines_detected"],
            "cnn_transcript": out["cnn_transcript"],
            "trocr_transcript": out["trocr_transcript"],
            "best_transcript": out.get("best_transcript", out.get("full_transcript", "")),
            "full_transcript": out["full_transcript"],
            "lines": out["lines"],
            "annotated_image_base64": out["annotated_image_base64"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/export")
async def export_transcript(
    transcript: str = Form(...),
    export_format: str = Form("txt")
):
    try:
        export_format = export_format.lower()
        if export_format == "json":
            data = {"project": "Efficient Text Recognition System", "transcript": transcript}
            content = json.dumps(data, indent=2)
            media_type = "application/json"
            filename = "ocr_transcript.json"
        else:
            header = "=========================================================\n" \
                     " MAJOR PROJECT CODE 42: EFFICIENT OCR TRANSCRIPT EXPORT  \n" \
                     "=========================================================\n\n"
            content = header + transcript
            media_type = "text/plain"
            filename = "ocr_transcript.txt"

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/analytics")
def get_analytics():
    history_path = "d:\\Major Project\\src\\models\\checkpoints\\training_history.json"
    history_data = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)

    figures_b64 = {}
    fig_dir = "d:\\Major Project\\docs\\figures"
    if os.path.exists(fig_dir):
        for fig_name in ["training_curves.png", "benchmark_comparison.png"]:
            p = os.path.join(fig_dir, fig_name)
            if os.path.exists(p):
                img = cv2.imread(p)
                figures_b64[fig_name.replace(".png", "")] = f"data:image/png;base64,{encode_img_to_b64(img)}"

    return JSONResponse({
        "status": "success",
        "training_history": history_data,
        "figures": figures_b64
    })

@app.get("/", response_class=HTMLResponse)
def get_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Web Demo UI Loading...</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
