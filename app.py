#!/usr/bin/env python3
"""
TripoSR Local Server — app.py
Wraps TripoSR to serve true AI-generated 3D meshes via a local Flask API.

Requirements: see requirements.txt
Run:          python app.py
"""

import os, io, sys, uuid, json, time, traceback, threading
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Expose-Headers"] = "X-Mesh-Stats,X-Processing-Time"
    return resp

@app.route("/convert",  methods=["OPTIONS"])
@app.route("/health",   methods=["OPTIONS"])
@app.route("/progress", methods=["OPTIONS"])
def _options(): return make_response("", 204)

# ── Model state ───────────────────────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()
_model_status = {"loaded": False, "loading": False, "error": None}

# Per-request progress tracking
_progress: dict[str, dict] = {}

def get_model():
    """Lazy-load TripoSR model (downloads ~2 GB on first run)."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        if _model_status["loading"]:
            return None

        _model_status["loading"] = True
        _model_status["error"]   = None
        log.info("Loading TripoSR model … (downloads ~2 GB on first call)")
        try:
            import torch
            from tsr.system import TSR

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Device: {device}")

            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model = model.to(device)
            model.eval()

            _model = {"model": model, "device": device}
            _model_status["loaded"]  = True
            _model_status["loading"] = False
            log.info("TripoSR model ready ✓")
            return _model
        except Exception as e:
            _model_status["error"]   = str(e)
            _model_status["loading"] = False
            log.error(f"Model load failed: {e}")
            raise

# Start loading in background at startup
def _bg_load():
    try:
        get_model()
    except Exception:
        pass

threading.Thread(target=_bg_load, daemon=True).start()

# ── Progress endpoint ─────────────────────────────────────────────────────────
@app.route("/progress/<job_id>")
def progress(job_id):
    data = _progress.get(job_id, {"step": "unknown", "pct": 0})
    return jsonify(data)

def set_progress(job_id, step, pct):
    _progress[job_id] = {"step": step, "pct": pct}

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    import torch
    return jsonify({
        "status":       "ok",
        "engine":       "triposr",
        "model_loaded": _model_status["loaded"],
        "model_loading":_model_status["loading"],
        "model_error":  _model_status["error"],
        "cuda":         torch.cuda.is_available() if _is_torch_available() else False,
        "device":       (_model or {}).get("device", "unknown"),
    })

def _is_torch_available():
    try: import torch; return True
    except ImportError: return False

# ── Foreground removal helper ─────────────────────────────────────────────────
def remove_background(img: Image.Image) -> Image.Image:
    """
    Attempt rembg background removal; fall back to centre-crop alpha masking.
    TripoSR works best with a clean white/transparent background.
    """
    try:
        from rembg import remove as rembg_remove
        log.info("Background removal via rembg")
        return rembg_remove(img)
    except ImportError:
        log.info("rembg not available — using centre-crop mask fallback")
        img = img.convert("RGBA")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        # Simple elliptical mask centred on image
        cy, cx = h/2, w/2
        Y, X = np.ogrid[:h, :w]
        mask = ((X-cx)**2/(cx*0.9)**2 + (Y-cy)**2/(cy*0.9)**2) <= 1.0
        arr[~mask, 3] = 0
        return Image.fromarray(arr.astype(np.uint8))

# ── Export helpers ────────────────────────────────────────────────────────────
def mesh_to_glb(mesh) -> bytes:
    """Export trimesh/mesh object to GLB bytes."""
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()

def mesh_to_obj_zip(mesh, stem: str) -> bytes:
    import zipfile
    obj_buf = io.BytesIO()
    mesh.export(obj_buf, file_type="obj")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.obj", obj_buf.getvalue())
    return zip_buf.getvalue()

def mesh_to_ply(mesh) -> bytes:
    buf = io.BytesIO()
    mesh.export(buf, file_type="ply")
    return buf.getvalue()

def mesh_to_stl(mesh) -> bytes:
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()

# ── Main convert route ────────────────────────────────────────────────────────
@app.route("/convert", methods=["POST"])
def convert():
    job_id = uuid.uuid4().hex[:12]
    set_progress(job_id, "Queued", 0)
    t0 = time.time()

    try:
        # ── validate inputs ──
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        fmt          = request.form.get("format", "glb").lower()
        remove_bg    = request.form.get("remove_bg", "true").lower() == "true"
        mc_threshold = float(request.form.get("mc_threshold", "25.0"))
        chunk_size   = int(request.form.get("chunk_size", "8192"))

        allowed_fmts = {"glb", "obj", "ply", "stl"}
        if fmt not in allowed_fmts:
            return jsonify({"error": f"Format must be one of {allowed_fmts}"}), 400

        # ── ensure model is loaded ──
        set_progress(job_id, "Loading model…", 5)
        ctx = get_model()
        if ctx is None:
            return jsonify({"error": "Model is still loading, please retry in a moment."}), 503

        import torch
        model  = ctx["model"]
        device = ctx["device"]

        # ── load & preprocess image ──
        set_progress(job_id, "Preprocessing image…", 15)
        img = Image.open(request.files["image"].stream).convert("RGBA")

        # Resize: TripoSR expects 512×512
        img = img.resize((512, 512), Image.LANCZOS)

        if remove_bg:
            set_progress(job_id, "Removing background…", 25)
            img = remove_background(img)

        # Composite onto white background for the model input
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img_rgb = bg.convert("RGB")

        # ── run TripoSR ──
        set_progress(job_id, "Running TripoSR inference…", 40)
        log.info(f"[{job_id}] Running inference, threshold={mc_threshold}, chunk={chunk_size}")

        with torch.no_grad():
            scene_codes = model([img_rgb], device=device)

        set_progress(job_id, "Extracting mesh…", 70)
        meshes = model.extract_mesh(
            scene_codes,
            resolution=256,
            threshold=mc_threshold,
            chunk_source_n=chunk_size,
        )
        mesh = meshes[0]

        set_progress(job_id, "Exporting…", 88)
        stem = f"triposr_{job_id}"

        if fmt == "glb":
            data, mime, fname = mesh_to_glb(mesh), "model/gltf-binary", f"{stem}.glb"
        elif fmt == "obj":
            data, mime, fname = mesh_to_obj_zip(mesh, stem), "application/zip", f"{stem}_obj.zip"
        elif fmt == "ply":
            data, mime, fname = mesh_to_ply(mesh), "application/octet-stream", f"{stem}.ply"
        elif fmt == "stl":
            data, mime, fname = mesh_to_stl(mesh), "application/octet-stream", f"{stem}.stl"

        elapsed = round(time.time() - t0, 2)
        stats = {
            "vertices":    len(mesh.vertices),
            "faces":       len(mesh.faces),
            "format":      fmt,
            "device":      device,
            "elapsed_sec": elapsed,
            "job_id":      job_id,
        }
        log.info(f"[{job_id}] Done in {elapsed}s — {stats['vertices']:,} verts, {stats['faces']:,} faces")

        set_progress(job_id, "Done", 100)
        resp = send_file(io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=fname)
        resp.headers["X-Mesh-Stats"]       = json.dumps(stats)
        resp.headers["X-Processing-Time"]  = str(elapsed)
        return resp

    except Exception as e:
        log.error(traceback.format_exc())
        set_progress(job_id, f"Error: {e}", -1)
        return jsonify({"error": str(e), "job_id": job_id}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"""
╔══════════════════════════════════════════════╗
║  DEPTH3D · TripoSR Local Server              ║
║  http://localhost:{port:<5}                       ║
║  Open index.html in your browser             ║
╚══════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
