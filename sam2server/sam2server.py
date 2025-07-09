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

# URL of the DIAM server
DIAM_SERVER_URL = "http://127.0.0.1:5001/receive_data"

# Set base data directories (pointing to the Data folder parallel to diamserver)
BASE_DIR = "../Data"
FRAMES_DIR = os.path.join(BASE_DIR, "frame")
MASK_DIR = os.path.join(BASE_DIR, "sam2mask")        
RESULTS_DIR = os.path.join(BASE_DIR, "sam2results")  

# Model configuration and checkpoint remain in the original SAM2 folder
MODEL_CFG = "../sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "../sam2/checkpoints/sam2.1_hiera_large.pt"

# Logging
LOG_DIR = os.path.join(BASE_DIR, "task_logs")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
os.makedirs(LOG_DIR, exist_ok=True)


# Task queue
task_queue = queue.Queue()
running_tasks = set()
MAX_CONCURRENT_TASKS = 2  # maximum number of concurrent tasks
queue_lock = threading.Lock()

log_lock = threading.Lock()
all_task_timings = []  # store each task's execution time (start, end)
received_tasks = []    # store all received tasks as (space_name, object_id, time)


def log(msg):
    """
    Append a log message with timestamp to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")


def rle_decode(rle, width, height):
    """
    Decode RLE (Run Length Encoding) into a NumPy 2D mask array.

    Args:
        rle (list of [value, count]): RLE-encoded mask.
        width (int): Image width.
        height (int): Image height.

    Returns:
        np.ndarray: 2D array of shape (height, width) containing 0/1.
    """
    mask_array = np.zeros((height * width), dtype=np.uint8)
    index = 0
    for value, count in rle:
        mask_array[index: index + count] = value
        index += count
    return mask_array.reshape((height, width))


def send_to_diamserver(space_name, object_id, description):
    """
    Asynchronously send `space_name` and `object_id` to DIAMserver.

    Args:
        space_name (str): Name of the space.
        object_id (str): Object ID.
        description (str): Optional description.
    """
    print("Sending data to DIAMserver:", space_name, object_id, description)

    payload = {
        "space_name": space_name,
        "object_id": object_id,
        "description": description
    }
    try:
        print("Payload sent to DIAMserver:", payload)
        response = requests.post(DIAM_SERVER_URL, json=payload, timeout=5)
        print("Response from DIAMserver:", response.json())  # debug output
    except requests.exceptions.RequestException as e:
        print("Error sending data to DIAMserver:", e)


def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list.

    Args:
        intervals (list of tuple): list of (start, end) datetime tuples.

    Returns:
        list of tuple: merged non-overlapping intervals.
    """
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
    """
    Write a summary of all tasks and their execution timeline to the log file.
    Includes:
    - Total number of tasks received.
    - Task details (space_name, object_id, time).
    - Merged task execution intervals.
    - Total runtime and idle time.
    """
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
    """
    Continuously monitor the task queue and ensure no more than MAX_CONCURRENT_TASKS
    are running at the same time.
    """
    while True:
        with queue_lock:
            if len(running_tasks) < MAX_CONCURRENT_TASKS and not task_queue.empty():
                task = task_queue.get()
                space_name = task["space_name"]
                object_id = task["object_id"]

                running_tasks.add(object_id)  # mark task as running
                print(f"Starting task {object_id}")

                # Start a new thread to execute the task
                threading.Thread(target=run_task, args=(task,)).start()

        time.sleep(1)  # prevent busy waiting


def run_task(task):
    """
    Execute a single task:
    - Decode mask from base64 RLE
    - Save mask as CSV
    - Run sam2mask.py and mask2bbox.py
    - Send result to DIAM server
    - Log timing and update task queue
    """
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
        frame_filename = f"frame_{ann_frame_idx:04d}.jpg"

        video_dir = os.path.join(FRAMES_DIR, space_name)
        mask_dir = os.path.join(MASK_DIR, space_name)
        mask_csv_path = os.path.join(mask_dir, f"{object_id}.csv")
        output_dir = os.path.join(RESULTS_DIR, space_name)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Decode base64 RLE
        decoded_mask = base64.b64decode(mask_base64).decode("utf-8")
        mask_data = json.loads(decoded_mask)
        width, height = mask_data["width"], mask_data["height"]
        rle_encoding = mask_data["encoding"]
        mask_array = rle_decode(rle_encoding, width, height)

        # Save mask as CSV
        print(f"Saving mask CSV to: {mask_csv_path}")
        with open(mask_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["width", "height"])
            writer.writerow([width, height])
            writer.writerows(mask_array)

        # Ensure CSV was saved correctly
        if not os.path.exists(mask_csv_path):
            return jsonify({"error": f"Mask CSV not found at {mask_csv_path}"}), 500

        # Run sam2mask.py
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

        # Run mask2bbox.py
        cmd_mask2bbox = [
            "python", "mask2bbox.py",
            "--space_name", space_name,
            "--object_id", object_id,
            "--description", description
        ]
        subprocess.run(cmd_mask2bbox, check=True)

        # Verify JSON result
        json_path = os.path.join(output_dir, object_id, "bbox_results.json")
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found for {object_id}")
            return
        print(f"Task {object_id} completed. Result stored at {json_path}")

        # Notify DIAM server asynchronously
        threading.Thread(target=send_to_diamserver, args=(space_name, object_id, description)).start()

    except Exception as e:
        print(f"Error processing task {object_id}: {e}")

    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log(f"[RUNNING END] {space_name} | {object_id} | {end_time} | duration: {duration:.1f}s")
        all_task_timings.append((start_time, end_time))

        # Remove task from running set
        with queue_lock:
            running_tasks.discard(object_id)  # ✅ use discard() to avoid KeyError
            print(f"Task {object_id} finished. Remaining running tasks: {len(running_tasks)}")

            # Check if we can start a new task
            if len(running_tasks) < MAX_CONCURRENT_TASKS and not task_queue.empty():
                next_task = task_queue.get()
                running_tasks.add(next_task["object_id"])
                threading.Thread(target=run_task, args=(next_task,)).start()
            if task_queue.empty() and not running_tasks:
                summarize_log()


SPACE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))

# Load Space data
class Space:
    def __init__(self, name, image, ply_file, json_files):
        """
        Represents a space with its associated data.

        Args:
            name (str): Name of the space.
            image (str): Filename of the first image found in the frame folder.
            ply_file (str): Filename of the first .ply file found.
            json_files (dict): Dictionary of parsed JSON files in bbox_results.
        """
        self.name = name
        self.image = image
        self.ply_file = ply_file
        self.json_files = json_files


def load_spaces():
    """
    Traverse the SPACE_FOLDER directory and dynamically load Space data.

    For each subfolder:
    - Find the first image (*.png or *.jpg) in the 'frame' folder.
    - Find the first .ply file in the space folder.
    - Parse all .json files in the 'bbox_results' folder.

    Returns:
        list of Space: list of loaded Space objects, sorted by name descending.
    """
    spaces = []

    if not os.path.exists(SPACE_FOLDER):
        print(f"Warning: {SPACE_FOLDER} does not exist.")
        return spaces

    for space_name in os.listdir(SPACE_FOLDER):
        space_path = os.path.join(SPACE_FOLDER, space_name)
        if os.path.isdir(space_path):
            # Image path: {SPACE_FOLDER}/{name}/frame/*.png or *.jpg
            frame_path = os.path.join(space_path, "frame")
            images = [
                f for f in os.listdir(frame_path)
                if f.endswith(('.png', '.jpg'))
            ] if os.path.exists(frame_path) else []
            image = images[0] if images else None  # take the first image if available

            # PLY file path: {SPACE_FOLDER}/{name}/*.ply
            ply_files = [
                f for f in os.listdir(space_path)
                if f.endswith('.ply')
            ]
            ply_file = ply_files[0] if ply_files else None  # take the first .ply file if available

            # JSON files path: {SPACE_FOLDER}/{name}/bbox_results/*.json
            bbox_results_path = os.path.join(space_path, "bbox_results")
            json_files = {}

            if os.path.exists(bbox_results_path):
                for json_file in os.listdir(bbox_results_path):
                    if json_file.endswith('.json'):
                        json_file_path = os.path.join(bbox_results_path, json_file)
                        try:
                            with open(json_file_path, 'r', encoding="utf-8") as f:
                                # parse JSON file content and store in dictionary
                                json_files[json_file] = json.load(f)
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: {json_file} in {space_name} is not a valid JSON file.")

            if image and ply_file and json_files:
                space = Space(space_name, image, ply_file, json_files)
                spaces.append(space)

    # sort spaces by name in descending order
    spaces.sort(key=lambda x: x.name, reverse=True)
    return spaces


@app.route('/')
def home():
    """
    Health check route.
    """
    return jsonify({"message": "Flask server is running!"})


@app.route('/process', methods=['POST'])
def process():
    """
    Receive a request from the frontend and add it to the task queue.
    """
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
            task_queue.put(data)  # add task to queue
            print(f"Task {object_id} added to queue. Queue size: {task_queue.qsize()}")
            # check if a task can be started immediately
            if len(running_tasks) < MAX_CONCURRENT_TASKS:
                next_task = task_queue.get()
                running_tasks.add(next_task["object_id"])
                threading.Thread(target=run_task, args=(next_task,)).start()

        def wait_for_task_completion():
            """
            Keep the HTTP connection open and wait until the task completes,
            then return the resulting JSON file.
            """
            output_dir = os.path.join(RESULTS_DIR, space_name)
            json_path = os.path.join(output_dir, object_id, "bbox_results.json")

            # wait until the task produces the JSON file
            while not os.path.exists(json_path):
                time.sleep(0.5)  # check every 500ms
                yield ""

            print(f"Task {object_id} completed. Sending file to client.")

            # return the JSON result
            with open(json_path, "r") as file:
                yield file.read()

        # use Response to keep the connection alive until the task completes
        return Response(wait_for_task_completion(), content_type='application/json')

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# API: Get the list of all Spaces
@app.route('/spaces', methods=['GET'])
def get_spaces():
    """
    Get the list of all Spaces, reloading data from disk on every request.
    """
    spaces = load_spaces()
    space_list = [
        {
            "name": space.name,
            "image": space.image,
            "ply_file": space.ply_file,  # include .ply file path
            "json_files": space.json_files
        }
        for space in spaces
    ]
    return jsonify(space_list)


# API: Get details of a specific Space
@app.route('/space_details/<space_name>', methods=['GET'])
def get_space_details(space_name):
    """
    Get details of a specific Space by name, reloading data from disk on every request.
    """
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


# API: Serve an image file from a Space's 'frame' folder
@app.route('/images/<space_name>/<image_name>', methods=['GET'])
def serve_image(space_name, image_name):
    """
    Serve an image file (.png/.jpg) from the 'frame' folder of a Space.
    """
    image_path = os.path.join(SPACE_FOLDER, space_name, "frame")
    if os.path.exists(os.path.join(image_path, image_name)):
        return send_from_directory(image_path, image_name)
    return jsonify({"error": "Image not found"}), 404


# API: Serve a .ply or .json file from a Space directory
@app.route('/files/<space_name>/<file_name>', methods=['GET'])
def serve_file(space_name, file_name):
    """
    Serve a .ply file (from Space root) or a .json file (from bbox_results folder).
    """
    ply_path = os.path.join(SPACE_FOLDER, space_name, file_name)
    json_path = os.path.join(SPACE_FOLDER, space_name, "bbox_results", file_name)

    if os.path.exists(ply_path):
        return send_from_directory(os.path.join(SPACE_FOLDER, space_name), file_name)
    elif os.path.exists(json_path):
        return send_from_directory(os.path.join(SPACE_FOLDER, space_name, "bbox_results"), file_name)

    return jsonify({"error": "File not found"}), 404


@app.route('/get_result', methods=['GET'])
def get_result():
    """
    Retrieve the final bbox_results.json file for a given space and object.
    """
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
    # Start the background task management thread
    threading.Thread(target=process_tasks, daemon=True).start()
    # Start the Flask server
    app.run(host='0.0.0.0', port=80, debug=True)
