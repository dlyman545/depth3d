#!/usr/bin/env python3
"""
DEPTH3D — Shared Meshy Proxy for Railway
Anyone can use this deployment by supplying their own Meshy API key.
The proxy never stores keys — they are passed per-request from the browser.
No MESHY_API_KEY env var needed on Railway (users bring their own keys).
"""

import os, io, time, json, base64, traceback, requests
from flask import Flask, request, jsonify, send_file, make_response
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

MESHY_BASE = "https://api.meshy.ai/openapi/v2"
MESHY_TEST = "msy_dummy_api_key_for_test_mode_12345678"

# ── CORS — open to all origins so any browser can call this ───────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]   = "*"
    r.headers["Access-Control-Allow-Headers"]  = "Content-Type"
    r.headers["Access-Control-Allow-Methods"]  = "GET, POST, OPTIONS"
    r.headers["Access-Control-Expose-Headers"] = "X-Mesh-Stats, X-Processing-Time"
    return r

@app.route("/convert", methods=["OPTIONS"])
@app.route("/health",  methods=["OPTIONS"])
@app.route("/balance", methods=["OPTIONS"])
def _opt(): return make_response("", 204)

def _resolve_key():
    """
    Key priority:
      1. 'api_key' field in POST form data  — user's own key sent from browser
      2. 'key' query param                  — for GET requests like /health?key=msy_xxx
      3. MESHY_TEST fallback               — returns sample result, uses 0 credits
    No server-side env var key. Credits always come from the user's own account.
    """
    return (request.form.get("api_key") or
            request.args.get("key") or
            MESHY_TEST)

def _meshy_headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# ── Health ─────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    key     = _resolve_key()
    is_test = (key == MESHY_TEST)
    result  = {"engine": "meshy-proxy", "test_mode": is_test, "shared": True}
    try:
        r = requests.get(f"{MESHY_BASE}/balance", headers=_meshy_headers(key), timeout=8)
        if r.ok:
            result["balance"] = r.json()
        else:
            result["balance_note"] = "key invalid or test mode"
    except Exception as e:
        result["balance_error"] = str(e)
    return jsonify(result)

# ── Balance ────────────────────────────────────────────────────────────────────
@app.route("/balance")
def balance():
    key = _resolve_key()
    try:
        r = requests.get(f"{MESHY_BASE}/balance", headers=_meshy_headers(key), timeout=8)
        return jsonify(r.json() if r.ok else {"error": r.text}), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Convert ────────────────────────────────────────────────────────────────────
@app.route("/convert", methods=["POST"])
def convert():
    t0  = time.time()
    key = _resolve_key()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    fmt        = request.form.get("format", "glb").lower()
    ai_model   = request.form.get("ai_model", "meshy-4")
    do_texture = request.form.get("texture", "true") == "true"

    file     = request.files["image"]
    raw      = file.read()
    ext      = (file.filename or "image.png").rsplit(".", 1)[-1].lower()
    mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}
    data_uri = f"data:{mime_map.get(ext,'image/png')};base64,{base64.b64encode(raw).decode()}"

    log.info(f"Request: model={ai_model} fmt={fmt} test={'yes' if key==MESHY_TEST else 'no'}")

    # Submit to Meshy
    try:
        r = requests.post(
            f"{MESHY_BASE}/image-to-3d",
            headers=_meshy_headers(key),
            json={"image_url": data_uri, "ai_model": ai_model,
                  "should_texture": do_texture, "enable_pbr": False},
            timeout=30
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach Meshy: {e}"}), 502

    if not r.ok:
        try:
            err_msg = r.json().get("message") or r.text
        except Exception:
            err_msg = r.text
        if r.status_code == 401:
            err_msg = "Invalid Meshy API key. Check your key at app.meshy.ai → Settings → API."
        elif r.status_code == 402:
            err_msg = "Not enough Meshy credits. Top up at app.meshy.ai."
        return jsonify({"error": err_msg, "status_code": r.status_code}), r.status_code

    task_id = r.json().get("result")
    if not task_id:
        return jsonify({"error": "No task ID returned by Meshy"}), 500

    log.info(f"Task: {task_id}")

    # Poll until complete
    deadline = time.time() + 300
    interval = 4

    while time.time() < deadline:
        time.sleep(interval)
        interval = min(interval + 2, 12)
        try:
            pr = requests.get(f"{MESHY_BASE}/image-to-3d/{task_id}",
                              headers=_meshy_headers(key), timeout=15)
        except requests.RequestException:
            continue
        if not pr.ok:
            continue

        data   = pr.json()
        status = data.get("status", "")
        log.info(f"  [{task_id}] {status} {data.get('progress',0)}%")

        if status == "SUCCEEDED":
            fmt_map  = {"glb":"glb","obj":"obj","stl":"stl","ply":"obj"}
            dl_url   = data.get("model_urls",{}).get(fmt_map.get(fmt,"glb")) or \
                       data.get("model_urls",{}).get("glb")
            if not dl_url:
                return jsonify({"error": "No download URL from Meshy"}), 500

            dl = requests.get(dl_url, timeout=60)
            if not dl.ok:
                return jsonify({"error": f"Download failed: HTTP {dl.status_code}"}), 502

            elapsed  = round(time.time() - t0, 2)
            ext_map  = {"glb":".glb","obj":"_obj.zip","stl":".stl","ply":".ply"}
            mime_out = {"glb":"model/gltf-binary","obj":"application/zip",
                        "stl":"application/octet-stream"}.get(fmt,"application/octet-stream")

            stats = {"task_id":task_id,"format":fmt,"ai_model":ai_model,
                     "elapsed_sec":elapsed,"thumbnail":data.get("thumbnail_url","")}

            log.info(f"Done {task_id} in {elapsed}s")
            resp = send_file(io.BytesIO(dl.content), mimetype=mime_out,
                             as_attachment=True, download_name=f"meshy_{task_id[:8]}{ext_map.get(fmt,'.glb')}")
            resp.headers["X-Mesh-Stats"]      = json.dumps(stats)
            resp.headers["X-Processing-Time"] = str(elapsed)
            return resp

        elif status in ("FAILED", "EXPIRED"):
            return jsonify({"error": data.get("task_error",{}).get("message","Task failed"),
                            "task_id": task_id}), 500

    return jsonify({"error": "Timed out after 5 minutes", "task_id": task_id}), 504

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"\n  DEPTH3D shared Meshy proxy on port {port}")
    print(f"  Users supply their own Meshy API keys — no server-side key needed\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
