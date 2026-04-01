#!/usr/bin/env python3
"""
DEPTH3D — TripoSR Local Backend (app_local.py)
No git clone required. Install via:  pip install git+https://github.com/VAST-AI-Research/TripoSR
Run:  python app_local.py
"""

import os, io, uuid, json, time, struct, zipfile, traceback, threading
import numpy as np
from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    r.headers["Access-Control-Expose-Headers"] = "X-Mesh-Stats,X-Processing-Time"
    return r

@app.route("/convert",  methods=["OPTIONS"])
@app.route("/health",   methods=["OPTIONS"])
def _opt(): return make_response("", 204)

# ── Model state ───────────────────────────────────────────────────────────────
_ctx  = None
_lock = threading.Lock()
_status = {"loaded": False, "loading": False, "error": None, "device": "?"}

def _load_model():
    global _ctx
    with _lock:
        if _ctx: return _ctx
        _status["loading"] = True
        try:
            import torch
            from tsr.system import TSR
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Loading TripoSR on {device} …")
            m = TSR.from_pretrained("stabilityai/TripoSR",
                                     config_name="config.yaml",
                                     weight_name="model.ckpt")
            m.to(device).eval()
            _ctx = {"model": m, "device": device}
            _status.update(loaded=True, loading=False, device=device)
            log.info("Model ready ✓")
        except Exception as e:
            _status.update(loading=False, error=str(e))
            raise
        return _ctx

threading.Thread(target=lambda: _load_model() or None, daemon=True).start()

# ── Background removal ────────────────────────────────────────────────────────
def remove_bg(img: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(img)
    except ImportError:
        img = img.convert("RGBA")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        Y, X = np.ogrid[:h, :w]
        mask = ((X - w/2)**2/(w*0.45)**2 + (Y - h/2)**2/(h*0.45)**2) <= 1.0
        arr[~mask, 3] = 0
        return Image.fromarray(arr.astype(np.uint8))

# ── Export helpers ────────────────────────────────────────────────────────────
def to_bytes(mesh, fmt):
    buf = io.BytesIO()
    if fmt == "glb":
        mesh.export(buf, file_type="glb"); return buf.getvalue(), "model/gltf-binary", ".glb"
    if fmt == "obj":
        mesh.export(buf, file_type="obj")
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.obj", buf.getvalue())
        return z.getvalue(), "application/zip", "_obj.zip"
    if fmt == "ply":
        mesh.export(buf, file_type="ply"); return buf.getvalue(), "application/octet-stream", ".ply"
    if fmt == "stl":
        mesh.export(buf, file_type="stl"); return buf.getvalue(), "application/octet-stream", ".stl"
    raise ValueError(f"Unknown format {fmt}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    try: import torch; cuda = torch.cuda.is_available()
    except ImportError: cuda = False
    return jsonify({**_status, "engine": "triposr-local", "cuda": cuda})

@app.route("/convert", methods=["POST"])
def convert():
    t0 = time.time()
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    fmt        = request.form.get("format", "glb").lower()
    do_rembg   = request.form.get("remove_bg", "true") == "true"
    threshold  = float(request.form.get("mc_threshold", "25"))
    chunk      = int(request.form.get("chunk_size", "8192"))
    if fmt not in ("glb","obj","ply","stl"):
        return jsonify({"error": "Bad format"}), 400
    try:
        ctx = _load_model()
        import torch
        model, device = ctx["model"], ctx["device"]
        img = Image.open(request.files["image"].stream).convert("RGBA").resize((512,512), Image.LANCZOS)
        if do_rembg: img = remove_bg(img)
        bg = Image.new("RGBA", (512,512), (255,255,255,255))
        bg.paste(img, mask=img.split()[3])
        img_rgb = bg.convert("RGB")
        with torch.no_grad():
            codes = model([img_rgb], device=device)
        meshes = model.extract_mesh(codes, resolution=256, threshold=threshold, chunk_source_n=chunk)
        mesh   = meshes[0]
        data, mime, ext = to_bytes(mesh, fmt)
        stem  = f"triposr_{uuid.uuid4().hex[:8]}"
        elapsed = round(time.time() - t0, 2)
        stats = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                 "format": fmt, "device": device, "elapsed_sec": elapsed}
        resp = send_file(io.BytesIO(data), mimetype=mime,
                         as_attachment=True, download_name=stem+ext)
        resp.headers["X-Mesh-Stats"]      = json.dumps(stats)
        resp.headers["X-Processing-Time"] = str(elapsed)
        return resp
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  ▸ TripoSR local server → http://localhost:{port}\n  ▸ Open index.html in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
