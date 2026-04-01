#!/usr/bin/env python3
"""
DEPTH3D — Meshy API Proxy
Deployable to Railway. Set MESHY_API_KEY in Railway environment variables.
"""

import os, io, time, json, uuid, base64, traceback, requests
from flask import Flask, request, jsonify, send_file, make_response
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

MESHY_BASE = "https://api.meshy.ai/openapi/v2"
MESHY_KEY  = os.environ.get("MESHY_API_KEY", "")
MESHY_TEST = "msy_dummy_api_key_for_test_mode_12345678"

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Meshy-Key"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Expose-Headers"]= "X-Mesh-Stats, X-Processing-Time"
    return r

@app.route("/convert", methods=["OPTIONS"])
@app.route("/health",  methods=["OPTIONS"])
@app.route("/balance", methods=["OPTIONS"])
def _opt(): return make_response("", 204)

def _key(req_key=None):
    return req_key or MESHY_KEY or MESHY_TEST

def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    key     = _key(request.args.get("key"))
    is_test = (key == MESHY_TEST)
    result  = {"engine": "meshy-proxy", "test_mode": is_test,
               "key_configured": bool(MESHY_KEY)}
    try:
        r = requests.get(f"{MESHY_BASE}/balance", headers=_headers(key), timeout=8)
        if r.ok:
            result["balance"] = r.json()
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

    fmt        = request.form.get("format", "glb").lower()
    ai_model   = request.form.get("ai_model", "meshy-4")
    do_texture = request.form.get("texture", "true") == "true"

    # Encode image as base64 data URI
    file     = request.files["image"]
    raw      = file.read()
    ext      = (file.filename or "image.png").rsplit(".", 1)[-1].lower()
    mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}
    data_uri = f"data:{mime_map.get(ext,'image/png')};base64,{base64.b64encode(raw).decode()}"

    log.info(f"Submitting: model={ai_model} texture={do_texture} key={'test' if key==MESHY_TEST else 'live'}")

    # Step 1 — create task
    try:
        r = requests.post(f"{MESHY_BASE}/image-to-3d", headers=_headers(key),
                          json={"image_url": data_uri, "ai_model": ai_model,
                                "should_texture": do_texture, "enable_pbr": False},
                          timeout=30)
    except requests.RequestException as e:
        return jsonify({"error": f"Network error: {e}"}), 502

    if not r.ok:
        err = r.json() if "application/json" in r.headers.get("content-type","") else {}
        return jsonify({"error": err.get("message", r.text), "code": r.status_code}), r.status_code

    task_id = r.json().get("result")
    if not task_id:
        return jsonify({"error": "No task_id returned", "raw": r.json()}), 500

    log.info(f"Task created: {task_id}")

    # Step 2 — poll
    deadline      = time.time() + 300
    poll_interval = 4

    while time.time() < deadline:
        time.sleep(poll_interval)
        poll_interval = min(poll_interval + 2, 12)

        try:
            pr = requests.get(f"{MESHY_BASE}/image-to-3d/{task_id}",
                              headers=_headers(key), timeout=15)
        except requests.RequestException:
            continue

        if not pr.ok:
            continue

        data   = pr.json()
        status = data.get("status", "")
        log.info(f"  [{task_id}] {status} {data.get('progress',0)}%")

        if status == "SUCCEEDED":
            model_urls = data.get("model_urls", {})
            fmt_map    = {"glb":"glb","obj":"obj","stl":"stl","ply":"obj"}
            dl_url     = model_urls.get(fmt_map.get(fmt,"glb")) or model_urls.get("glb")

            if not dl_url:
                return jsonify({"error": "No download URL", "model_urls": model_urls}), 500

            dl = requests.get(dl_url, timeout=60)
            if not dl.ok:
                return jsonify({"error": f"Download failed: {dl.status_code}"}), 502

            elapsed  = round(time.time() - t0, 2)
            ext_map  = {"glb":".glb","obj":"_obj.zip","stl":".stl","ply":".ply"}
            mime_out = {"glb":"model/gltf-binary","obj":"application/zip",
                        "stl":"application/octet-stream"}.get(fmt,"application/octet-stream")
            filename = f"meshy_{task_id[:8]}{ext_map.get(fmt,'.glb')}"

            stats = {"task_id": task_id, "format": fmt, "engine": "meshy",
                     "ai_model": ai_model, "elapsed_sec": elapsed,
                     "thumbnail": data.get("thumbnail_url","")}

            resp = send_file(io.BytesIO(dl.content), mimetype=mime_out,
                             as_attachment=True, download_name=filename)
            resp.headers["X-Mesh-Stats"]      = json.dumps(stats)
            resp.headers["X-Processing-Time"] = str(elapsed)
            return resp

        elif status in ("FAILED", "EXPIRED"):
            msg = data.get("task_error", {}).get("message", "Task failed")
            return jsonify({"error": msg, "task_id": task_id}), 500

    return jsonify({"error": "Timed out (5 min)", "task_id": task_id}), 504

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"  ▸ Meshy proxy running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
