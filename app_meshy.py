#!/usr/bin/env python3
"""
DEPTH3D — Meshy API Proxy (app_meshy.py)
No GPU, no PyTorch, no cloning. Just an API key from meshy.ai (free tier available).
Run:  MESHY_API_KEY=your_key python app_meshy.py
Docs: https://docs.meshy.ai/en/api/image-to-3d
"""

import os, io, time, json, uuid, base64, traceback, requests
from flask import Flask, request, jsonify, send_file, make_response
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

MESHY_BASE  = "https://api.meshy.ai/openapi/v2"
MESHY_KEY   = os.environ.get("MESHY_API_KEY", "")
# Meshy's official test key — returns sample results, uses 0 credits
MESHY_TEST  = "msy_dummy_api_key_for_test_mode_12345678"

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    r.headers["Access-Control-Expose-Headers"] = "X-Mesh-Stats,X-Processing-Time,X-Credits-Used"
    return r

@app.route("/convert",  methods=["OPTIONS"])
@app.route("/health",   methods=["OPTIONS"])
@app.route("/balance",  methods=["OPTIONS"])
def _opt(): return make_response("", 204)

def _key(req_key=None):
    """Resolve which API key to use: request header > env var > test key."""
    return req_key or MESHY_KEY or MESHY_TEST

def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# ── Health / balance ──────────────────────────────────────────────────────────
@app.route("/health")
def health():
    key   = _key(request.args.get("key"))
    is_test = (key == MESHY_TEST)
    result = {"engine": "meshy-api", "test_mode": is_test,
              "key_configured": bool(key and key != MESHY_TEST)}
    try:
        r = requests.get(f"{MESHY_BASE}/balance", headers=_headers(key), timeout=8)
        if r.ok:
            result["balance"] = r.json()
        else:
            result["balance_error"] = r.text[:120]
    except Exception as e:
        result["balance_error"] = str(e)
    return jsonify(result)

@app.route("/balance")
def balance():
    key = _key(request.args.get("key") or request.headers.get("X-Meshy-Key"))
    try:
        r = requests.get(f"{MESHY_BASE}/balance", headers=_headers(key), timeout=8)
        return jsonify(r.json() if r.ok else {"error": r.text}), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Convert ───────────────────────────────────────────────────────────────────
@app.route("/convert", methods=["POST"])
def convert():
    t0  = time.time()
    key = _key(request.form.get("api_key") or request.headers.get("X-Meshy-Key"))

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    fmt         = request.form.get("format", "glb").lower()
    ai_model    = request.form.get("ai_model", "meshy-4")   # meshy-4 = 5 credits, meshy-6 = 20
    do_texture  = request.form.get("texture", "true") == "true"
    symmetry    = request.form.get("symmetry", "auto")

    # Encode image as base64 data URI
    file     = request.files["image"]
    raw      = file.read()
    ext      = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "png"
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_typ = mime_map.get(ext, "image/png")
    b64      = base64.b64encode(raw).decode()
    data_uri = f"data:{mime_typ};base64,{b64}"

    log.info(f"Submitting to Meshy: model={ai_model}, texture={do_texture}, key={'test' if key==MESHY_TEST else 'live'}")

    # ── Step 1: create task ──
    payload = {
        "image_url":     data_uri,
        "ai_model":      ai_model,
        "should_texture": do_texture,
        "symmetry_mode": symmetry,
        "enable_pbr":    False,
    }
    try:
        r = requests.post(f"{MESHY_BASE}/image-to-3d", headers=_headers(key),
                          json=payload, timeout=30)
    except requests.RequestException as e:
        return jsonify({"error": f"Network error: {e}"}), 502

    if not r.ok:
        err = r.json() if r.headers.get("content-type","").startswith("application/json") else {"error": r.text}
        return jsonify({"error": err.get("message", r.text), "code": r.status_code}), r.status_code

    task_id = r.json().get("result")
    if not task_id:
        return jsonify({"error": "No task_id returned", "raw": r.json()}), 500

    log.info(f"Task created: {task_id}")

    # ── Step 2: poll for completion ──
    deadline = time.time() + 300   # 5-minute timeout
    poll_interval = 4
    last_status = None

    while time.time() < deadline:
        time.sleep(poll_interval)
        poll_interval = min(poll_interval + 2, 12)   # back off up to 12s

        try:
            pr = requests.get(f"{MESHY_BASE}/image-to-3d/{task_id}",
                              headers=_headers(key), timeout=15)
        except requests.RequestException:
            continue

        if not pr.ok:
            continue

        data   = pr.json()
        status = data.get("status", "")
        prog   = data.get("progress", 0)
        log.info(f"  [{task_id}] {status} {prog}%")

        if status == last_status and status not in ("IN_PROGRESS", "PENDING"):
            pass
        last_status = status

        if status == "SUCCEEDED":
            # Pick download URL for requested format
            model_urls  = data.get("model_urls", {})
            texture_urls = data.get("texture_urls", [])

            # Meshy returns glb, fbx, obj, usdz, stl, blend
            fmt_map = {"glb": "glb", "obj": "obj", "stl": "stl", "ply": "obj"}
            meshy_fmt = fmt_map.get(fmt, "glb")
            dl_url = model_urls.get(meshy_fmt) or model_urls.get("glb")

            if not dl_url:
                return jsonify({"error": "No download URL in response", "model_urls": model_urls}), 500

            log.info(f"Downloading {meshy_fmt} from {dl_url}")
            dl = requests.get(dl_url, timeout=60)
            if not dl.ok:
                return jsonify({"error": f"Download failed: {dl.status_code}"}), 502

            file_bytes = dl.content
            elapsed    = round(time.time() - t0, 2)
            stats      = {
                "task_id":     task_id,
                "format":      fmt,
                "engine":      "meshy-api",
                "ai_model":    ai_model,
                "elapsed_sec": elapsed,
                "thumbnail":   data.get("thumbnail_url", ""),
            }

            ext_map  = {"glb": ".glb", "obj": "_obj.zip", "stl": ".stl", "ply": ".ply"}
            dl_name  = f"meshy_{task_id[:8]}{ext_map.get(fmt, '.glb')}"
            out_mime = {"glb": "model/gltf-binary", "obj": "application/zip",
                        "stl": "application/octet-stream"}.get(fmt, "application/octet-stream")

            # If user wants PLY, convert obj to ply via trimesh if available
            if fmt == "ply":
                try:
                    import trimesh, io as _io
                    tm = trimesh.load(_io.BytesIO(file_bytes), file_type="obj")
                    pbuf = _io.BytesIO(); tm.export(pbuf, file_type="ply")
                    file_bytes = pbuf.getvalue()
                    out_mime   = "application/octet-stream"
                    dl_name    = f"meshy_{task_id[:8]}.ply"
                except Exception:
                    pass   # fall back to sending OBJ

            resp = send_file(io.BytesIO(file_bytes), mimetype=out_mime,
                             as_attachment=True, download_name=dl_name)
            resp.headers["X-Mesh-Stats"]      = json.dumps(stats)
            resp.headers["X-Processing-Time"] = str(elapsed)
            return resp

        elif status in ("FAILED", "EXPIRED"):
            msg = data.get("task_error", {}).get("message", "Task failed")
            return jsonify({"error": msg, "task_id": task_id, "status": status}), 500

    return jsonify({"error": "Timed out waiting for Meshy task", "task_id": task_id}), 504

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    key_info = f"live key ({MESHY_KEY[:8]}…)" if MESHY_KEY else f"test key (0 credits)"
    print(f"""
  ▸ Meshy API proxy → http://localhost:{port}
  ▸ Using: {key_info}
  ▸ Get a free key at https://app.meshy.ai → Settings → API
  ▸ Or set env var: MESHY_API_KEY=msy_xxxx python app_meshy.py
  ▸ Open index.html in your browser
""")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
