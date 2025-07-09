from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from flask_cors import CORS
import os
import subprocess
import numpy as np
import json
import base64
import csv
import requests
import threading
import queue
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

DIAM_SERVER_URL = "http://127.0.0.1:5001/receive_data"

# 设置本地路径
BASE_DIR = "../Data"
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
MASK_DIR = os.path.join(BASE_DIR, "mask")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_CFG = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"

LOG_DIR = os.path.join(BASE_DIR, "task_logs")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
os.makedirs(LOG_DIR, exist_ok=True)

# 任务队列
task_queue = queue.Queue()
running_tasks = set()
MAX_CONCURRENT_TASKS = 2  # 最多同时运行两个任务
queue_lock = threading.Lock()

log_lock = threading.Lock()
all_task_timings = []  # 存储每个任务的运行时间段（start, end）
received_tasks = []  # 存储所有接收到的任务（space_name, object_id, time）

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")

def rle_decode(rle, width, height):
    """
    解码 RLE 并转换为 NumPy 2D 数组
    :param rle: 运行长度编码 (list of [value, count])
    :param width: 图像宽度
    :param height: 图像高度
    :return: (height, width) 形状的 NumPy 数组，包含 0/1
    """
    mask_array = np.zeros((height * width), dtype=np.uint8)
    index = 0
    for value, count in rle:
        mask_array[index: index + count] = value
        index += count
    return mask_array.reshape((height, width))

def send_to_diamserver(space_name, object_id, description):
    """异步发送 space_name 和 object_id 给 DIAMserver"""

    print("Sending data to DIAMserver:", space_name, object_id, description)

    payload = {
        "space_name": space_name,
        "object_id": object_id,
        "description": description
    }
    try:
        print("Payload sent to DIAMserver:", payload)
        response = requests.post(DIAM_SERVER_URL, json=payload, timeout=5)
        print("Response from DIAMserver:", response.json())  # 调试输出
    except requests.exceptions.RequestException as e:
        print("Error sending data to DIAMserver:", e)

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def summarize_log():
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n================= Summary =================\n")
            f.write(f"Total Tasks Received: {len(received_tasks)}\n\n")
            for idx, (s, o, t) in enumerate(received_tasks, 1):
                f.write(f"{idx}. space_name: {s}, object_id: {o}, received_at: {t}\n")

            f.write("\nTask Execution Timeline:\n")
            merged = merge_intervals(all_task_timings)
            total_runtime = sum((e - s).total_seconds() for s, e in merged)
            full_span = (merged[-1][1] - merged[0][0]).total_seconds() if merged else 0
            idle_time = full_span - total_runtime

            f.write(f"\nTotal Runtime: {total_runtime:.1f} seconds\n")
            f.write(f"Idle Time: {idle_time:.1f} seconds\n")
            f.write("===========================================\n\n")

def process_tasks():
    """持续检查任务队列，并确保最多只有 2 个任务同时运行"""
    while True:
        with queue_lock:
            if len(running_tasks) < MAX_CONCURRENT_TASKS and not task_queue.empty():
                task = task_queue.get()
                space_name = task["space_name"]
                object_id = task["object_id"]

                running_tasks.add(object_id)  # 标记任务正在运行
                print(f"Starting task {object_id}")

                # 启动新线程执行任务
                threading.Thread(target=run_task, args=(task,)).start()

        time.sleep(1)  # 避免 CPU 过度占用

def run_task(task):
    """执行单个任务"""
    try:
        space_name = task["space_name"]
        frame_name = task["frame_name"]
        object_id = task["object_id"]
        mask_base64 = task["mask"]
        description = task["description"]
        start_time = datetime.now()
        log(f"[RUNNING START] {space_name} | {object_id} | {start_time}")
        print("description:", description, type(description))


        print(f"Processing task: {task}")
        ann_frame_idx = int(frame_name.split('_')[1].split('.')[0])
        #ann_frame_idx = int(frame_name)
        frame_filename = f"frame_{ann_frame_idx:04d}.jpg"

        video_dir = os.path.join(FRAMES_DIR, space_name)
        mask_dir = os.path.join(MASK_DIR, space_name)
        mask_csv_path = os.path.join(mask_dir, f"{object_id}.csv")
        output_dir = os.path.join(RESULTS_DIR, space_name)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # 解码 Base64 RLE
        decoded_mask = base64.b64decode(mask_base64).decode("utf-8")
        mask_data = json.loads(decoded_mask)
        width, height = mask_data["width"], mask_data["height"]
        rle_encoding = mask_data["encoding"]
        mask_array = rle_decode(rle_encoding, width, height)

        # 保存 CSV
        print(f"Saving mask CSV to: {mask_csv_path}")
        with open(mask_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["width", "height"])
            writer.writerow([width, height])
            writer.writerows(mask_array)
        # 确保 CSV 文件正确生成
        if not os.path.exists(mask_csv_path):
            return jsonify({"error": f"Mask CSV not found at {mask_csv_path}"}), 500

        # 运行 sam2mask.py
        cmd_sam2mask = [
            "python", "sam2mask.py",
            "--video_dir", video_dir,
            "--mask_csv", mask_csv_path,
            "--model_cfg", MODEL_CFG,
            "--checkpoint", CHECKPOINT,
            "--class_id", object_id,
            "--ann_frame_idx", str(ann_frame_idx),
            "--output_dir", output_dir
        ]
        subprocess.run(cmd_sam2mask, check=True)

        # 运行 mask2bbox.py
        cmd_mask2bbox = [
            "python", "mask2bbox.py",
            "--space_name", space_name,
            "--object_id", object_id,
            "--description", description
        ]
        subprocess.run(cmd_mask2bbox, check=True)

        # 获取 JSON 结果文件
        json_path = os.path.join(output_dir, object_id, "bbox_results.json")
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found for {object_id}")
            return
        print(f"Task {object_id} completed. Result stored at {json_path}")
        
        # **异步通知 DIAM 服务器**
        threading.Thread(target=send_to_diamserver, args=(space_name, object_id, description)).start()


    except Exception as e:
        print(f"Error processing task {object_id}: {e}")

    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log(f"[RUNNING END] {space_name} | {object_id} | {end_time} | duration: {duration:.1f}s")
        all_task_timings.append((start_time, end_time))

        # 任务完成，从运行任务集合中移除
        with queue_lock:
            running_tasks.discard(object_id)  # ✅ 使用 `discard()` 避免 KeyError
            print(f"Task {object_id} finished. Remaining running tasks: {len(running_tasks)}")

            # **检查队列，看看是否可以启动新的任务**
            if len(running_tasks) < MAX_CONCURRENT_TASKS and not task_queue.empty():
                next_task = task_queue.get()
                running_tasks.add(next_task["object_id"])
                threading.Thread(target=run_task, args=(next_task,)).start()
            if task_queue.empty() and not running_tasks:
                summarize_log()





SPACE_FOLDER = r"C:\Users\xiasu\Desktop\Ruiqi_DIAM\Data"

# 加载 Space 数据
class Space:
    def __init__(self, name, image, ply_file, json_files):
        self.name = name
        self.image = image
        self.ply_file = ply_file
        self.json_files = json_files

def load_spaces():
    """遍历 SPACE_FOLDER 目录，动态加载 Space 数据"""
    spaces = []

    if not os.path.exists(SPACE_FOLDER):
        print(f"Warning: {SPACE_FOLDER} does not exist.")
        return spaces

    for space_name in os.listdir(SPACE_FOLDER):
        space_path = os.path.join(SPACE_FOLDER, space_name)
        if os.path.isdir(space_path):
            # 图像路径：{SPACE_FOLDER}/{name}/frame/*.png 或 *.jpg
            frame_path = os.path.join(space_path, "frame")
            images = [f for f in os.listdir(frame_path) if f.endswith(('.png', '.jpg'))] if os.path.exists(frame_path) else []
            image = images[0] if images else None  # 取第一张图片

            # PLY 文件路径：{SPACE_FOLDER}/{name}/*.ply
            ply_files = [f for f in os.listdir(space_path) if f.endswith('.ply')]
            ply_file = ply_files[0] if ply_files else None  # 取第一个 .ply 文件

            # JSON 文件路径：{SPACE_FOLDER}/{name}/bbox_results/*.json
            bbox_results_path = os.path.join(space_path, "bbox_results")
            json_files = {}

            if os.path.exists(bbox_results_path):
                for json_file in os.listdir(bbox_results_path):
                    if json_file.endswith('.json'):
                        json_file_path = os.path.join(bbox_results_path, json_file)
                        try:
                            with open(json_file_path, 'r', encoding="utf-8") as f:
                                json_files[json_file] = json.load(f)  # ✅ 解析 JSON 文件内容并存入字典
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: {json_file} in {space_name} is not a valid JSON file.")

            if image and ply_file and json_files:
                space = Space(space_name, image, ply_file, json_files)
                spaces.append(space)

    spaces.sort(key=lambda x: x.name, reverse=True)  # 按名称降序排序
    return spaces




@app.route('/')
def home():
    return jsonify({"message": "Flask server is running!"})

@app.route('/process', methods=['POST'])
def process():
    """接收前端请求并加入任务队列"""
    try:
        data = request.get_json()
        space_name = data.get("space_name")
        frame_name = data.get("frame_name")
        object_id = data.get("object_id")
        mask_base64 = data.get("mask")
        description = data.get("description", {})

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        received_tasks.append((space_name, object_id, timestamp))
        log(f"[RECEIVED] {space_name} | {object_id} | {timestamp}")

        if any(val is None or val == "" for val in [space_name, frame_name, object_id, mask_base64]):
            return jsonify({"error": "Missing required parameters"}), 400
        print(f"Task {object_id} received and added to queue.")

        with queue_lock:
            task_queue.put(data)  # 把任务加入队列
            print(f"Task {object_id} added to queue. Queue size: {task_queue.qsize()}")
            # **检查是否可以立即执行任务**
            if len(running_tasks) < MAX_CONCURRENT_TASKS:
                next_task = task_queue.get()
                running_tasks.add(next_task["object_id"])
                threading.Thread(target=run_task, args=(next_task,)).start()
        
        def wait_for_task_completion():
            """保持 HTTP 连接，等待任务完成后返回 JSON 文件"""
            output_dir = os.path.join(RESULTS_DIR, space_name)
            json_path = os.path.join(output_dir, object_id, "bbox_results.json")

            # **等待任务完成**
            while not os.path.exists(json_path):
                time.sleep(0.5)  # 每 500ms 检查一次是否生成 JSON 文件
                yield "" 

            print(f"Task {object_id} completed. Sending file to client.")

            # **返回 JSON 结果**
            with open(json_path, "r") as file:
                yield file.read()

        # **使用 `Response` 保持连接，直到任务完成**
        return Response(wait_for_task_completion(), content_type='application/json')


    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# API: 获取所有 Space 列表
@app.route('/spaces', methods=['GET'])
def get_spaces():
    """获取 Space 列表，确保每次请求时重新加载数据"""
    spaces = load_spaces()
    space_list = [
        {
            "name": space.name,
            "image": space.image,
            "ply_file": space.ply_file,  # ✅ 确保返回 ply 文件路径
            "json_files": space.json_files
        }
        for space in spaces
    ]
    return jsonify(space_list)

# API: 获取指定 Space 的详细信息
@app.route('/space_details/<space_name>', methods=['GET'])
def get_space_details(space_name):
    """获取特定 Space 详细信息，每次调用时重新读取本地数据"""
    spaces = load_spaces()
    for space in spaces:
        if space.name == space_name:
            return jsonify({
                "name": space.name,
                "image": space.image,
                "ply_file": space.ply_file,
                "json": space.json_files
            })

    return jsonify({"error": "Space not found"}), 404

# API: 访问 Space 目录中的图片（frame 目录）
@app.route('/images/<space_name>/<image_name>', methods=['GET'])
def serve_image(space_name, image_name):
    """提供 Space 相关图片（frame 目录下的 .png/.jpg 文件）"""
    image_path = os.path.join(SPACE_FOLDER, space_name, "frame")
    if os.path.exists(os.path.join(image_path, image_name)):
        return send_from_directory(image_path, image_name)
    return jsonify({"error": "Image not found"}), 404

# API: 访问 Space 目录中的 .ply 和 .json 文件
@app.route('/files/<space_name>/<file_name>', methods=['GET'])
def serve_file(space_name, file_name):
    """提供 Space 相关文件（.ply 位于 Space 根目录，.json 位于 bbox_results 目录）"""
    ply_path = os.path.join(SPACE_FOLDER, space_name, file_name)
    json_path = os.path.join(SPACE_FOLDER, space_name, "bbox_results", file_name)

    if os.path.exists(ply_path):
        return send_from_directory(os.path.join(SPACE_FOLDER, space_name), file_name)
    elif os.path.exists(json_path):
        return send_from_directory(os.path.join(SPACE_FOLDER, space_name, "bbox_results"), file_name)

    return jsonify({"error": "File not found"}), 404


@app.route('/get_result', methods=['GET'])
def get_result():
    space_name = request.args.get("space_name")
    object_id = request.args.get("object_id")
    
    if not all([space_name, object_id]):
        return jsonify({"error": "Missing parameters"}), 400
    
    output_dir = os.path.join(RESULTS_DIR, space_name)
    json_path = os.path.join(output_dir, "bbox_results.json")
    
    if not os.path.exists(json_path):
        return jsonify({"error": "No JSON file found"}), 404
    
    return send_file(json_path, as_attachment=True)

if __name__ == '__main__':
    threading.Thread(target=process_tasks, daemon=True).start()  # 启动任务管理线程
    app.run(host='0.0.0.0', port=80, debug=True)
