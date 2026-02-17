from flask import Flask, render_template, Response, request, jsonify, send_file
from video_stream import camera
import os

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    if not camera.system_ready:
        return "<script>alert('System not initialized. Please select device first.'); window.location.href='/';</script>"
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init_system():
    data = request.json
    device = data.get('device', 'cpu')
    success = camera.initialize_system(device)
    if success:
         return jsonify({"status": "success"})
    else:
         return jsonify({"status": "error", "message": "Initialization failed"}), 500

def gen(camera):
    while True:
        frame = camera.get_frame()

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
             # If camera is stopped or fails, sleep briefly
             import time
             time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_enrollment', methods=['POST'])
def start_enrollment():
    data = request.json
    name = data.get('name')
    user_id = data.get('id')
    
    if not name or not user_id:
        return jsonify({"status": "error", "message": "Name and ID required"}), 400
    
    camera.start_enrollment(name, user_id)
    return jsonify({"status": "success", "message": "Enrollment started"})

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    camera.start_attendance()
    return jsonify({"status": "success", "message": "Attendance started"})

@app.route('/stop', methods=['POST'])
def stop():
    camera.stop_mode()
    return jsonify({"status": "success", "message": "Stopped"})

@app.route('/get_attendance_data')
def get_attendance_data():
    students = camera.get_marked_students()
    return jsonify({
        "students": students,
        "mode": camera.mode
    })

@app.route('/download_csv')
def download_csv():
    # Ensure file exists
    csv_path = "data/attendance.csv"
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return "No attendance data found", 404

@app.route('/download_session_csv')
def download_session_csv():
    csv_content = camera.get_session_csv()
    
    # Create a response with CSV content
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=session_attendance.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
