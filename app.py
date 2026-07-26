from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import os
import base64
import numpy as np
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
# Membuka izin komunikasi lintas domain (dari DomaiNesia ke sistem lokal Ngrok)
CORS(app)

model = YOLO('best.pt')

counted_ids = set()
stats = {"helmet": 0, "no_helmet": 0, "total": 0}
violation_logs = []
last_frame = None
is_paused = False  
paused_base64_cache = None # Menambahkan variabel untuk menyimpan base64 saat jeda agar CPU tidak bekerja keras

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

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
    
    if is_paused:
        if paused_base64_cache is not None:
            return jsonify({"image": paused_base64_cache})
        return jsonify({"status": "paused"})

    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data"}), 400

    try:
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            
            for track_id, class_id in zip(track_ids, class_ids):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    stats["total"] += 1
                    
                    class_name = model.names[class_id].lower()
                    
                    if "no" in class_name or "tanpa" in class_name or "without" in class_name:
                        stats["no_helmet"] += 1
                        
                        # FITUR BARU 1: Format Hari, Tanggal, dan Waktu Bahasa Indonesia
                        hari_dict = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                        idx_hari = datetime.now().weekday()
                        tgl_str = datetime.now().strftime("%d-%m-%Y")
                        waktu_str = datetime.now().strftime("%H:%M:%S")
                        waktu_lengkap = f"{hari_dict[idx_hari]}, {tgl_str} | {waktu_str}"
                        
                        annotated_snapshot = results[0].plot()
                        saved_path = save_automatic_snapshot(annotated_snapshot, "tanpa_helm", track_id)
                        
                        violation_logs.insert(0, {
                            "time": waktu_lengkap, 
                            "status": "Tanpa Helm",
                            "snapshot": saved_path
                        })
                        
                        if len(violation_logs) > 15:
                            violation_logs.pop()
                            
                    elif "helmet" in class_name or "helm" in class_name:
                        stats["helmet"] += 1

        annotated_frame = results[0].plot()
        last_frame = annotated_frame 
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        result_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        paused_base64_cache = result_b64
        
        return jsonify({"image": result_b64})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": "Failed to process frame"}), 500

# FITUR BARU 3: Rute untuk Reset Data Harian
@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global stats, violation_logs, counted_ids
    stats = {"helmet": 0, "no_helmet": 0, "total": 0}
    violation_logs = []
    counted_ids = set() # Membersihkan ID wajah/objek yang sudah terekam agar esok hari dihitung dari awal
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
    return jsonify({"stats": stats, "logs": violation_logs, "is_paused": is_paused})

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
        return jsonify({"status": "success", "message": f"Gambar disimpan di folder: {filename}"})
    return jsonify({"status": "error", "message": "Gagal menangkap layar."})

# FITUR BARU WAJIB: Route agar website bisa mengambil dan menampilkan foto dari folder backend
@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory('snapshots', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)