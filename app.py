#!/usr/bin/env python3
"""
Image to 3D Asset Converter — Flask Backend
Deploy on Railway. Converts 2D images to 3D via depth estimation + mesh generation.
"""

import os, io, uuid, json, struct, zipfile, traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Expose-Headers'] = 'X-Mesh-Stats'
    return response

@app.route('/convert', methods=['OPTIONS'])
@app.route('/health',  methods=['OPTIONS'])
def options(): return make_response('', 204)

# ── Depth Estimation ──────────────────────────────────────────────────────────
def estimate_depth_map(img_array):
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    lum = (0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2]) / 255.0
    try:
        from scipy.ndimage import convolve, gaussian_filter
        k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
        edges = np.abs(convolve(lum.astype(np.float32), k))
        edges = gaussian_filter(edges, sigma=2)
        smooth_lum = gaussian_filter(lum, sigma=3)
    except ImportError:
        edges = np.zeros_like(lum)
        smooth_lum = lum
    h, w = lum.shape
    y_grad = np.linspace(0.2, 0.8, h)[:, None] * np.ones((1, w))
    depth = 0.4*smooth_lum + 0.3*(1.0 - edges/(edges.max()+1e-8)) + 0.3*y_grad
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)

def depth_to_mesh(depth, img, scale_z=0.4, subsample=2):
    h, w = depth.shape
    ys = np.arange(0, h, subsample); xs = np.arange(0, w, subsample)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    sh, sw = gy.shape
    nx=(gx/(w-1))*2-1; ny=(gy/(h-1))*2-1; nz=depth[gy,gx]*scale_z
    vertices = np.stack([nx,-ny,nz], axis=-1).reshape(-1,3).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 4: img = img[:,:,:3]
    colors = img[gy,gx].reshape(-1,3).astype(np.uint8)
    def idx(r,c): return r*sw+c
    faces = []
    for r in range(sh-1):
        for c in range(sw-1):
            a,b,cc,d = idx(r,c),idx(r,c+1),idx(r+1,c),idx(r+1,c+1)
            faces.append([a,b,cc]); faces.append([b,d,cc])
    return vertices, colors, np.array(faces, dtype=np.int32)

# ── Exporters ─────────────────────────────────────────────────────────────────
def export_ply(vertices, colors, faces):
    nv,nf=len(vertices),len(faces)
    header=(f"ply\nformat binary_little_endian 1.0\nelement vertex {nv}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    vdata=bytearray()
    for v,c in zip(vertices,colors): vdata+=struct.pack('<fff',*v)+struct.pack('<BBB',*c)
    fdata=bytearray()
    for f in faces: fdata+=struct.pack('<B',3)+struct.pack('<iii',*f)
    return header+bytes(vdata)+bytes(fdata)

def export_obj(vertices, colors, faces, stem):
    lines=["# img3d",f"mtllib {stem}.mtl",""]
    for v,c in zip(vertices,colors):
        lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]/255:.4f} {c[1]/255:.4f} {c[2]/255:.4f}")
    lines.append("")
    for f in faces: lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.obj","\n".join(lines).encode())
        zf.writestr(f"{stem}.mtl",b"newmtl default\nKa 1 1 1\nKd 1 1 1\n")
    return buf.getvalue()

def export_stl(vertices, faces):
    buf=bytearray(80)+struct.pack('<I',len(faces))
    for f in faces:
        v0,v1,v2=vertices[f[0]],vertices[f[1]],vertices[f[2]]
        e1,e2=v1-v0,v2-v0; n=np.cross(e1,e2); ln=np.linalg.norm(n)
        n=n/ln if ln>0 else n
        buf+=struct.pack('<fff',*n)+struct.pack('<fff',*v0)+struct.pack('<fff',*v1)+struct.pack('<fff',*v2)+struct.pack('<H',0)
    return bytes(buf)

def export_gltf(vertices, colors, faces, stem):
    def pad(b): return b+b'\x00'*(-len(b)%4)
    vb=pad(vertices.astype(np.float32).tobytes()); vl=len(vb)
    cb=pad((colors.astype(np.float32)/255.0).tobytes()); cl=len(cb)
    fb=pad(faces.astype(np.uint32).tobytes()); fl=len(fb)
    bin_data=vb+cb+fb
    vmin=vertices.min(axis=0).tolist(); vmax=vertices.max(axis=0).tolist()
    gltf={"asset":{"version":"2.0","generator":"img3d"},"scene":0,
          "scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
          "meshes":[{"name":stem,"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"indices":2,"mode":4}]}],
          "accessors":[
              {"bufferView":0,"componentType":5126,"count":len(vertices),"type":"VEC3","min":vmin,"max":vmax},
              {"bufferView":1,"componentType":5126,"count":len(vertices),"type":"VEC3"},
              {"bufferView":2,"componentType":5125,"count":len(faces)*3,"type":"SCALAR"}],
          "bufferViews":[
              {"buffer":0,"byteOffset":0,"byteLength":vl,"target":34962},
              {"buffer":0,"byteOffset":vl,"byteLength":cl,"target":34962},
              {"buffer":0,"byteOffset":vl+cl,"byteLength":fl,"target":34963}],
          "buffers":[{"byteLength":len(bin_data),"uri":f"{stem}.bin"}]}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.gltf",json.dumps(gltf,indent=2))
        zf.writestr(f"{stem}.bin",bin_data)
    return buf.getvalue()

def export_xyz(vertices, colors):
    return "\n".join(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]} {c[1]} {c[2]}"
                     for v,c in zip(vertices,colors)).encode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health(): return jsonify({"status":"ok","engine":"python"})

@app.route('/convert', methods=['POST'])
def convert():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"No image file provided"}), 400
        file=request.files['image']
        fmt=request.form.get('format','ply').lower()
        quality=request.form.get('quality','medium').lower()
        scale_z=float(request.form.get('depth_scale','0.4'))
        subsample={'low':4,'medium':2,'high':1}.get(quality,2)

        img_pil=Image.open(file.stream).convert('RGBA')
        w,h=img_pil.size
        if max(w,h)>512:
            s=512/max(w,h); img_pil=img_pil.resize((int(w*s),int(h*s)),Image.LANCZOS)
        img_array=np.array(img_pil); img_rgb=img_array[:,:,:3]

        depth=estimate_depth_map(img_array)
        vertices,colors,faces=depth_to_mesh(depth,img_rgb,scale_z=scale_z,subsample=subsample)
        stem=f"model_{uuid.uuid4().hex[:8]}"

        dispatch={
            'ply': (lambda:(export_ply(vertices,colors,faces),'application/octet-stream',f"{stem}.ply")),
            'obj': (lambda:(export_obj(vertices,colors,faces,stem),'application/zip',f"{stem}_obj.zip")),
            'stl': (lambda:(export_stl(vertices,faces),'application/octet-stream',f"{stem}.stl")),
            'gltf':(lambda:(export_gltf(vertices,colors,faces,stem),'application/zip',f"{stem}_gltf.zip")),
            'xyz': (lambda:(export_xyz(vertices,colors),'text/plain',f"{stem}.xyz")),
        }
        if fmt not in dispatch: return jsonify({"error":f"Unknown format: {fmt}"}),400
        data,mime,filename=dispatch[fmt]()

        stats={"vertices":len(vertices),"faces":len(faces),"image_size":list(img_pil.size),"format":fmt}
        resp=send_file(io.BytesIO(data),mimetype=mime,as_attachment=True,download_name=filename)
        resp.headers['X-Mesh-Stats']=json.dumps(stats)
        return resp
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error":str(e)}),500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    print(f"▸ img3d running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV')=='development')
