from flask import Flask, request, jsonify, send_file
import os
import sys
import threading
import queue
import time
from datetime import datetime

# 确保可以导入 edit_segdep.py 和 function_cast.py
sys.path.append(os.path.abspath(".."))

from edit_segdep import combine_segmentation_and_depth
from function_cast import process_depth_segmentation
from cal_box import process_and_save_bbox
from return_all_bbox import save_merged_data

app = Flask(__name__)

# 创建任务队列
task_queue = queue.Queue()
all_data_received = False

# 用于记录总处理时长
total_runtime = 0
runtime_lock = threading.Lock()

def log_task_to_file(space_name, object_id, duration=None, total_runtime=None, success=True, reason=None):
    """记录任务信息到日志文件：包括成功与失败的任务"""

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
    """ 循环处理任务队列，确保任务按顺序执行 """
    global all_data_received, total_runtime

    while True:
        space_name, object_id, description = task_queue.get()  # 获取任务
        try:
            print(f"Processing task: space_name={space_name}, object_id={object_id}")
            start_time = time.time()

            # 设置路径
            depth_folder = f'../Data/{space_name}/depth'
            segmentation_folder = f'../SAM2/Data/results/{space_name}/{object_id}'
            segdep_folder = f'../Data/{space_name}/segdep/{object_id}'

            # 检查 segmentation_folder 是否存在且有内容
            if not os.path.exists(segmentation_folder) or \
                not any(f.endswith('.npy') for f in os.listdir(segmentation_folder)):
                    print(f"[SKIP] No .npy segmentation data for {space_name} - {object_id}. Skipping task.")
                    log_task_to_file(space_name, object_id, success=False, reason="No .npy segmentation data found.")
                    task_queue.task_done()
                    continue



            # 确保输出文件夹存在
            os.makedirs(segdep_folder, exist_ok=True)

            # 运行 combine_segmentation_and_depth
            print(f"Running combine_segmentation_and_depth for {space_name} - {object_id}...")
            combine_segmentation_and_depth(depth_folder, segmentation_folder, segdep_folder)
            print(f"Segmentation and depth combined successfully for {space_name} - {object_id}")

            # 运行 process_depth_segmentation
            print(f"Running process_depth_segmentation for {space_name} - {object_id}...")
            process_depth_segmentation(space_name, object_id)
            print(f"Processing casting completed for {space_name} - {object_id}")

            # 运行 process_and_save_bbox
            print(f"Running process_and_save_bbox for {space_name} - {object_id}...")
            process_and_save_bbox(space_name, object_id, description)
            print(f"Bounding box processing completed for {space_name} - {object_id}")

            # 计算耗时并记录
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

        task_queue.task_done()  # 标记任务完成

# 启动后台线程处理任务
task_thread = threading.Thread(target=process_task, daemon=True)
task_thread.start()

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """ 处理接收到的数据：
        - 若为 `space_name` 和 `object_id`，加入任务队列。
        - 若请求 `Give me all the json for {space_name}`，等待任务执行完毕再返回 JSON 文件。
    """
    global all_data_received

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request"}), 400

        # 处理 JSON 请求
        if isinstance(data, str) and data.startswith("Give me all the json for "):
            space_name = data.replace("Give me all the json for ", "").strip()
            
            if not space_name:
                return jsonify({"error": "Missing space_name"}), 400

            print(f"Received request for merged JSON: {space_name}")

            # 标记所有数据已收到，不会再有新的数据
            all_data_received = True

            # 等待所有任务完成
            while not task_queue.empty():
                print(f"Waiting for remaining tasks... {task_queue.qsize()} tasks left.")
                time.sleep(5)  # 每 2 秒检查一次队列是否清空

            # 生成 JSON 文件
            json_file_path = save_merged_data(space_name)

            if not os.path.exists(json_file_path):
                return jsonify({"error": "JSON file not found"}), 500

            print(f"All tasks done. Sending final JSON file for {space_name}.")
            return send_file(json_file_path, as_attachment=True)
        

        # 处理正常的 space_name 和 object_id
        space_name = data.get("space_name")
        object_id = data.get("object_id")
        description = data.get("description")

        if not space_name or not object_id:
            return jsonify({"error": "Missing parameters"}), 400

        # 记录任务
        task_queue.put((space_name, object_id, description))
        print(f"Task added: space_name={space_name}, object_id={object_id}, description = {description}")

        return jsonify({"status": "success", "message": "Data received successfully"}), 200

    except Exception as e:
        print(f"Error receiving data: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 运行 Flask 服务器，监听 5001 端口
    app.run(host='0.0.0.0', port=5001, debug=True)
