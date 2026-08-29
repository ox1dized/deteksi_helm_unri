from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import cv2
import os
import base64
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import psutil
import subprocess
import platform
import torch

app = Flask(__name__)
# Membuka izin komunikasi lintas domain (dari DomaiNesia ke sistem lokal Ngrok)
CORS(app)

model = YOLO('best.pt')
last_capture_time = 0  # Menyimpan waktu terakhir kamera menjepret pelanggar

counted_ids = set()
violation_ids = set() # Untuk melacak siapa saja yang sudah difoto (agar tidak difoto berkali-kali)
compliant_ids = set() # Untuk melacak siapa yang awalnya ditebak berhelm
last_detection_time = 0  # <--- TAMBAHKAN BARIS INI
stats = {"helmet": 0, "no_helmet": 0, "total": 0}
violation_logs = []
last_frame = None
is_paused = False  
paused_base64_cache = None # Menambahkan variabel untuk menyimpan base64 saat jeda agar CPU tidak bekerja keras

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

# Add runtime/stat globals
fps = 0.0
last_frame_time = None
last_latency = 0.0
last_inference_time = 0.0
last_cpu_percent = 0.0
last_ram_percent = 0.0
last_ram_used_mb = 0.0
last_gpu_info = {"name": None, "memory_total_mb": None, "memory_used_mb": None, "utilization": None}

def save_automatic_snapshot(frame, label, track_id):
    date_folder = datetime.now().strftime('%Y-%m-%d')
    target_dir = os.path.join('snapshots', date_folder)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    time_str = datetime.now().strftime('%H%M%S')
    
    # Perbaikan penggabungan nama path agar aman di semua OS
    filename = f"{label}_{time_str}_id{track_id}.jpg"
    filepath = os.path.join(target_dir, filename)
    
    cv2.imwrite(filepath, frame)
    
    # Mengembalikan path relatif agar bisa dikirim ke frontend untuk ditampilkan di tabel log
    return f"{date_folder}/{filename}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_frame, stats, violation_logs, is_paused, paused_base64_cache
    global counted_ids, violation_ids, compliant_ids, last_capture_time, last_detection_time 
    global fps, last_frame_time, last_latency, last_inference_time

    if is_paused:
        if paused_base64_cache is not None:
            return jsonify({"image": paused_base64_cache})
        return jsonify({"status": "paused"})

    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data"}), 400

    try:
        frame_process_start = time.time()   # start total latency timer

        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Measure inference time separately
        infer_start = time.time()
        results = model.track(frame, persist=True, verbose=False, conf=0.3, iou=0.45)
        infer_end = time.time()
        inference_time = infer_end - infer_start

        current_time = time.time()

        # Compute per-frame latency (total processing until here will include subsequent steps too)
        # We'll compute total after finishing encoding below.
        # Update FPS using time since last processed frame (instantaneous)
        if last_frame_time is None:
            frame_interval = None
        else:
            frame_interval = frame_process_start - last_frame_time

        if frame_interval and frame_interval > 0:
            inst_fps = 1.0 / frame_interval
            # Exponential moving average for stability
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps
        elif fps == 0.0:
            fps = 0.0

        last_frame_time = frame_process_start

        # PENGHAPUSAN MEMORI OTOMATIS: 
        if (current_time - last_detection_time) > 3.0:
            counted_ids.clear()
            violation_ids.clear()
            compliant_ids.clear()

        # JIKA ADA OBJEK TERDETEKSI DI LAYAR
        if len(results[0].boxes) > 0:
            last_detection_time = current_time # Reset timer jalanan kosong
            boxes = results[0].boxes
            class_ids = boxes.cls.int().cpu().tolist()
            
            # ==============================================================
            # SKENARIO 1: OBJEK NORMAL (Berhasil mendapatkan ID dari YOLO)
            # ==============================================================
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                
                for track_id, class_id in zip(track_ids, class_ids):
                    class_name = model.names[class_id].lower()
                    is_violation = "no" in class_name or "tanpa" in class_name or "without" in class_name
                    is_compliant = "helmet" in class_name or "helm" in class_name
                    
                    trigger_snapshot = False

                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        stats["total"] += 1
                        
                        if is_violation:
                            stats["no_helmet"] += 1
                            violation_ids.add(track_id)
                            trigger_snapshot = True
                        elif is_compliant:
                            stats["helmet"] += 1
                            compliant_ids.add(track_id)
                            
                    else:
                        if is_violation and track_id not in violation_ids:
                            if track_id in compliant_ids:
                                stats["helmet"] -= 1
                                compliant_ids.remove(track_id)
                            stats["no_helmet"] += 1
                            violation_ids.add(track_id)
                            trigger_snapshot = True
                            
                    if trigger_snapshot:
                        hari_dict = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                        waktu_lengkap = f"{hari_dict[datetime.now().weekday()]}, {datetime.now().strftime('%d-%m-%Y | %H:%M:%S')}"
                        
                        annotated_snapshot = results[0].plot()
                        saved_path = save_automatic_snapshot(annotated_snapshot, "tanpa_helm", track_id)
                        
                        violation_logs.insert(0, {"time": waktu_lengkap, "status": "Tanpa Helm", "snapshot": saved_path})
                        if len(violation_logs) > 15: violation_logs.pop()

            # ==============================================================
            # SKENARIO 2: DARURAT (Motor terlalu cepat, ID gagal terbaca)
            # ==============================================================
            else:
                for class_id in class_ids:
                    class_name = model.names[class_id].lower()
                    is_violation = "no" in class_name or "tanpa" in class_name or "without" in class_name
                    is_compliant = "helmet" in class_name or "helm" in class_name
                    
                    # Gunakan cooldown 1.5 detik agar tidak spam
                    if (current_time - last_capture_time) > 1.5:
                        
                        # ---> PERBAIKAN: Angka statistik kini ikut bertambah di skenario darurat
                        stats["total"] += 1
                        
                        if is_violation:
                            stats["no_helmet"] += 1
                            
                            # Eksekusi Jepretan
                            hari_dict = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                            waktu_lengkap = f"{hari_dict[datetime.now().weekday()]}, {datetime.now().strftime('%d-%m-%Y | %H:%M:%S')}"
                            
                            annotated_snapshot = results[0].plot()
                            saved_path = save_automatic_snapshot(annotated_snapshot, "tanpa_helm", int(current_time))
                            
                            violation_logs.insert(0, {"time": waktu_lengkap, "status": "Tanpa Helm", "snapshot": saved_path})
                            if len(violation_logs) > 15: violation_logs.pop()
                            
                        elif is_compliant:
                            stats["helmet"] += 1
                            
                        last_capture_time = current_time 
                        break # Cukup hitung 1 objek paling jelas per frame darurat

        annotated_frame = results[0].plot()
        last_frame = annotated_frame 

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        result_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        paused_base64_cache = result_b64

        frame_process_end = time.time()
        total_latency = frame_process_end - frame_process_start

        # update globals for admin metrics
        last_latency = total_latency
        last_inference_time = inference_time

        # collect system stats
        system = _collect_system_stats()

        # Simpan metrik ke file log (append)
        try:
            with open("metrics.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | fps={round(fps,2)} | latency_s={round(last_latency,3)} | inference_s={round(last_inference_time,3)} | cpu={system.get('cpu_percent')}% | ram={system.get('ram_percent')}% | gpu={system.get('gpu')}\n")
        except Exception:
            pass

        return jsonify({
            "image": result_b64,
            "metrics": {
                "fps": round(fps, 2),
                "latency_s": round(last_latency, 3),
                "inference_s": round(last_inference_time, 3),
                "cpu_percent": last_cpu_percent,
                "ram_percent": last_ram_percent,
                "ram_used_mb": last_ram_used_mb,
                "gpu": last_gpu_info
            }
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": "Failed to process frame"}), 500

# FITUR BARU 3: Rute untuk Reset Data Harian
@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global stats, violation_logs, counted_ids, violation_ids, compliant_ids
    stats = {"helmet": 0, "no_helmet": 0, "total": 0}
    violation_logs = []
    counted_ids = set() 
    violation_ids = set() 
    compliant_ids = set()
    return jsonify({"status": "success", "message": "Sistem direset ke nol."})

# FITUR BARU 4: Rute untuk Ekspor Rekapan ke TXT
@app.route('/export_log')
def export_log():
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Menyusun isi dokumen teks
    content = f"REKAPITULASI PENGAWASAN HELM\n"
    content += f"Tanggal: {date_str}\n"
    content += "="*40 + "\n"
    content += f"Total Kendaraan Terdeteksi  : {stats['total']}\n"
    content += f"Mematuhi Aturan (Berhelm)   : {stats['helmet']}\n"
    content += f"Pelanggaran (Tanpa Helm)    : {stats['no_helmet']}\n"
    content += "="*40 + "\n\n"
    
    content += "LOG KEJADIAN TERAKHIR (Max 15):\n"
    content += "-"*40 + "\n"
    if not violation_logs:
        content += "Tidak ada data pelanggaran.\n"
    else:
        for log in violation_logs:
            content += f"Waktu : {log['time']}\nStatus: {log['status']}\n\n"
            
    # Mengirimkan file teks untuk diunduh otomatis
    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=Rekap_Helm_{date_str}.txt"}
    )

@app.route('/get_stats')
def get_stats():
    system = _collect_system_stats()
    return jsonify({
        "stats": stats,
        "logs": violation_logs,
        "is_paused": is_paused,
        "metrics": {
            "fps": round(fps, 2),
            "last_frame_latency_s": round(last_latency, 3),
            "last_inference_s": round(last_inference_time, 3),
            "cpu_percent": system.get("cpu_percent"),
            "ram_percent": system.get("ram_percent"),
            "ram_used_mb": system.get("ram_used_mb"),
            "gpu": system.get("gpu")
        },
        "server": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_cuda_available": torch.cuda.is_available()
        }
    })

@app.route('/toggle_camera/<action>')
def toggle_camera(action):
    global is_paused
    if action == "pause":
        is_paused = True
        return jsonify({"status": "success"})
    elif action == "resume":
        is_paused = False
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route('/snapshot')
def snapshot():
    global last_frame
    if last_frame is not None:
        filename = f"snapshots/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, last_frame)
        # Kirim file langsung ke browser HP untuk diunduh otomatis
        return send_file(filename, as_attachment=True)
        # return jsonify({"status": "success", "message": f"Gambar disimpan di folder: {filename}"})
    return jsonify({"status": "error", "message": "Gagal menangkap layar."})

# FITUR BARU WAJIB: Route agar website bisa mengambil dan menampilkan foto dari folder backend
@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory('snapshots', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def _query_gpu_via_nvidia_smi():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL).decode().strip()
        # Example line: "GeForce GTX 1080, 8119, 1234, 12"
        name, mem_total, mem_used, util = [x.strip() for x in out.split(",")]
        return {"name": name,
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "utilization": int(util)}
    except Exception:
        return None

def _collect_system_stats():
    global last_cpu_percent, last_ram_percent, last_ram_used_mb, last_gpu_info
    try:
        last_cpu_percent = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        last_ram_percent = vm.percent
        last_ram_used_mb = int((vm.total - vm.available) / (1024 * 1024))
    except Exception:
        last_cpu_percent = last_ram_percent = last_ram_used_mb = None

    # GPU info: try nvidia-smi, fallback to torch.cuda
    gpu_info = _query_gpu_via_nvidia_smi()
    if gpu_info is None and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            gpu_info = {
                "name": torch.cuda.get_device_name(idx),
                "memory_total_mb": int(torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)),
                "memory_used_mb": int(torch.cuda.memory_allocated(idx) / (1024 * 1024)),
                "utilization": None
            }
        except Exception:
            gpu_info = None

    if gpu_info:
        last_gpu_info = gpu_info
    return {
        "cpu_percent": last_cpu_percent,
        "ram_percent": last_ram_percent,
        "ram_used_mb": last_ram_used_mb,
        "gpu": last_gpu_info
    }