from flask import Flask, request, jsonify, send_file
import os
import sys
import threading
import queue
import time
from datetime import datetime

# Ensure edit_segdep.py and function_cast.py can be imported
sys.path.append(os.path.abspath(".."))

from edit_segdep import combine_segmentation_and_depth
from function_cast import process_depth_segmentation
from cal_box import process_and_save_bbox
from return_all_bbox import save_merged_data

app = Flask(__name__)

# Create a task queue
task_queue = queue.Queue()
all_data_received = False

# Used to record the total processing time
total_runtime = 0
runtime_lock = threading.Lock()


def log_task_to_file(space_name, object_id, duration=None, total_runtime=None, success=True, reason=None):
    """
    Log task information to a file, including both successful and failed tasks.
    """
    log_dir = os.path.join("..", "Data", space_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"DIAMserver_{space_name}_log.txt")
    with open(log_path, "a") as f:
        f.write("==============================\n")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Task Time  : {timestamp}\n")
        f.write(f"Space Name : {space_name}\n")
        f.write(f"Object ID  : {object_id}\n")

        if success:
            f.write("Status     : SUCCESS\n")
            f.write(f"Duration   : {duration:.2f} seconds\n")
            f.write(f"--- Total Processing Time So Far: {total_runtime:.2f} seconds ---\n")
        else:
            f.write("Status     : FAILED\n")
            f.write(f"Reason     : {reason}\n")

        f.write("==============================\n\n")


def process_task():
    """
    Continuously process tasks from the queue, ensuring tasks are executed in order.
    """
    global all_data_received, total_runtime

    while True:
        space_name, object_id, description = task_queue.get()  # fetch a task from the queue
        try:
            print(f"Processing task: space_name={space_name}, object_id={object_id}")
            start_time = time.time()

            # Set up paths
            depth_folder = f'../Data/{space_name}/depth'
            segmentation_folder = f'../SAM2/Data/results/{space_name}/{object_id}'
            segdep_folder = f'../Data/{space_name}/segdep/{object_id}'

            # Check if segmentation folder exists and contains .npy files
            if not os.path.exists(segmentation_folder) or \
                not any(f.endswith('.npy') for f in os.listdir(segmentation_folder)):
                    print(f"[SKIP] No .npy segmentation data for {space_name} - {object_id}. Skipping task.")
                    log_task_to_file(space_name, object_id, success=False, reason="No .npy segmentation data found.")
                    task_queue.task_done()
                    continue

            # Ensure output folder exists
            os.makedirs(segdep_folder, exist_ok=True)

            # Run combine_segmentation_and_depth
            print(f"Running combine_segmentation_and_depth for {space_name} - {object_id}...")
            combine_segmentation_and_depth(depth_folder, segmentation_folder, segdep_folder)
            print(f"Segmentation and depth combined successfully for {space_name} - {object_id}")

            # Run process_depth_segmentation
            print(f"Running process_depth_segmentation for {space_name} - {object_id}...")
            process_depth_segmentation(space_name, object_id)
            print(f"Processing casting completed for {space_name} - {object_id}")

            # Run process_and_save_bbox
            print(f"Running process_and_save_bbox for {space_name} - {object_id}...")
            process_and_save_bbox(space_name, object_id, description)
            print(f"Bounding box processing completed for {space_name} - {object_id}")

            # Compute task duration and update total runtime
            end_time = time.time()
            duration = end_time - start_time

            with runtime_lock:
                total_runtime += duration

            print(f"[END] Finished task: {space_name} - {object_id} | Time: {duration:.2f}s")
            print(f"[INFO] Total processing time so far: {total_runtime:.2f}s\n")

            log_task_to_file(space_name, object_id, duration, total_runtime)

        except Exception as e:
            print(f"Error processing {space_name} - {object_id}: {str(e)}")
            log_task_to_file(space_name, object_id, success=False, reason=str(e))

        task_queue.task_done()  # mark task as done

# Start a background thread to process tasks
task_thread = threading.Thread(target=process_task, daemon=True)
task_thread.start()


@app.route('/receive_data', methods=['POST'])
def receive_data():
    """
    Handle incoming data:
    - If JSON contains `space_name` and `object_id`, add it to the task queue.
    - If the request is 'Give me all the json for {space_name}', wait for tasks to finish and return the final JSON.
    """
    global all_data_received

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request"}), 400

        # Handle request for final merged JSON
        if isinstance(data, str) and data.startswith("Give me all the json for "):
            space_name = data.replace("Give me all the json for ", "").strip()
            
            if not space_name:
                return jsonify({"error": "Missing space_name"}), 400

            print(f"Received request for merged JSON: {space_name}")

            # Mark that no more data will be sent
            all_data_received = True

            # Wait until all tasks in the queue are finished
            while not task_queue.empty():
                print(f"Waiting for remaining tasks... {task_queue.qsize()} tasks left.")
                time.sleep(5)  # check every 5 seconds

            # Generate the merged JSON file
            json_file_path = save_merged_data(space_name)

            if not os.path.exists(json_file_path):
                return jsonify({"error": "JSON file not found"}), 500

            print(f"All tasks done. Sending final JSON file for {space_name}.")
            return send_file(json_file_path, as_attachment=True)

        # Handle normal task with space_name and object_id
        space_name = data.get("space_name")
        object_id = data.get("object_id")
        description = data.get("description")

        if not space_name or not object_id:
            return jsonify({"error": "Missing parameters"}), 400

        # Add task to queue
        task_queue.put((space_name, object_id, description))
        print(f"Task added: space_name={space_name}, object_id={object_id}, description={description}")

        return jsonify({"status": "success", "message": "Data received successfully"}), 200

    except Exception as e:
        print(f"Error receiving data: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
