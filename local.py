from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import PlainTextResponse
import uvicorn, os, zipfile, subprocess, json, math, numpy as np
import requests, shutil
import io
import cv2 

app = FastAPI()
UPLOAD_DIR = "uploads"
COLMAP = r"C:\\Colmap\\COLMAP.bat"
COLAB_NERF_API = "https://0d3d-34-10-59-43.ngrok-free.app/nerf_input/"  # API

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def convert_to_ngp(cameras_txt, images_txt, img_folder, out_json):
    def parse_cameras_txt(path):
        cameras = {}
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == "":
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width = float(parts[2])
                height = float(parts[3])
                params = list(map(float, parts[4:]))
                cameras[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
        return cameras

    def parse_images_txt(path):
        images = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith('#') or line.strip() == "":
                    i += 1
                    continue
                parts = line.strip().split()
                if len(parts) < 10:
                    i += 1
                    continue
                image_name = os.path.basename(parts[9])
                images[image_name] = {
                    "qvec": list(map(float, parts[1:5])),
                    "tvec": list(map(float, parts[5:8])),
                    "camera_id": int(parts[8])
                }
                i += 2
        return images

    def create_transform_matrix(qvec, tvec):
        R = qvec2rotmat(qvec)
        t = np.array(tvec).reshape((3, 1))
        c2w = np.concatenate([R.T, -R.T @ t], axis=1)
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w[0:3, 1:3] *= -1
        return c2w

    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)

    first_cam = list(cameras.values())[0]
    if first_cam["model"] != "SIMPLE_RADIAL":
        raise NotImplementedError("Hiện tại chỉ hỗ trợ SIMPLE_RADIAL")

    f, cx, cy, _ = first_cam["params"]
    width = first_cam["width"]
    height = first_cam["height"]
    angle_x = 2 * math.atan(width / (2 * f))
    angle_y = 2 * math.atan(height / (2 * f))

    frames = []
    for name, data in images.items():
        image_path = os.path.join(img_folder, name)
        if not os.path.isfile(image_path):
            print(f"Bỏ qua ảnh thiếu: {name}")
            continue
        c2w = create_transform_matrix(data["qvec"], data["tvec"])
        frames.append({
            "file_path": f"images/{name}",
            "transform_matrix": c2w.tolist()
        })

    transforms = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": f,
        "fl_y": f,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "aabb_scale": 4,
        "frames": frames
    }

    with open(out_json, 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"transforms.json saved at {out_json}")

@app.post("/run_colmap/", response_class=PlainTextResponse)
async def run_colmap(file: UploadFile, scene_name: str = Form(...)):
    try:
        # vị trí
        scene_dir = os.path.join(UPLOAD_DIR, scene_name)
        img_dir = os.path.join(scene_dir, "images")
        sparse_dir = os.path.join(scene_dir, "sparse")
        db_path = os.path.join(scene_dir, "database.db")

        shutil.rmtree(scene_dir, ignore_errors=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)

        # lưu và giải nén
        zip_path = os.path.join(scene_dir, file.filename)
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(img_dir)


        # COLMAP
        def run_step(cmd):
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n[STDERR]\n{result.stderr}")

        run_step([COLMAP, "feature_extractor",
                  "--database_path", db_path,
                  "--image_path", img_dir,
                  "--SiftExtraction.use_gpu", "0",
                  "--log_to_stderr=true"])

        run_step([COLMAP, "exhaustive_matcher",
                  "--database_path", db_path,
                  "--log_to_stderr=true"])

        run_step([COLMAP, "mapper",
                  "--database_path", db_path,
                  "--image_path", img_dir,
                  "--output_path", sparse_dir,
                  "--log_to_stderr=true"])

        model_txt_dir = os.path.join(sparse_dir, "0")
        if not os.path.isdir(model_txt_dir):
            raise RuntimeError("COLMAP không tạo được thư mục sparse/0")

        run_step([COLMAP, "model_converter",
                  "--input_path", model_txt_dir,
                  "--output_path", model_txt_dir,
                  "--output_type", "TXT"])

        cameras_txt = os.path.join(model_txt_dir, "cameras.txt")
        images_txt = os.path.join(model_txt_dir, "images.txt")
        output_json = os.path.join(scene_dir, "transforms.json")

        convert_to_ngp(cameras_txt, images_txt, img_dir, output_json)

        # Gửi zip + json sang Colab
        with open(zip_path, 'rb') as zf, open(output_json, 'rb') as jf:
            zip_bytes = zf.read()
            json_bytes = jf.read()

            files = {
    "zip_file": ("input.zip", io.BytesIO(zip_bytes), "application/zip"),
    "json_file": ("transforms.json", io.BytesIO(json_bytes), "application/json")
}
            payload = {"scene_name": scene_name}

            r = requests.post(COLAB_NERF_API, files=files, data=payload, timeout=(20,300))
            return f"Đã gửi transforms.json và ảnh lên Colab.\nPhản hồi từ Colab: {r.status_code} - {r.text}"

    except Exception as e:
        import traceback
        return f"Lỗi:\n{traceback.format_exc()}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
