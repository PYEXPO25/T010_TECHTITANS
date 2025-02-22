from flask import Flask, render_template, jsonify,request,render_template
from detector import Monitor  # Your OpenCV monitoring script

app = Flask(__name__)
monitor = Monitor()

@app.route("/",methods=['GET','POST'])
def login():        
    return render_template("login.html")

@app.route("/exam",methods=['post'])
def exam():
    if request.method == "POST":

        return render_template("monitor.html")

@app.route("/start_camera")
def start_camera():
    return jsonify({"status": "Camera Started"})

@app.route("/stop_camera")
def stop_camera():
    monitor.stop_monitoring()
    return jsonify({"status": "Camera Stopped"})

@app.route("/check_suspicious")
def check_suspicious():
    monitor.start_monitoring()
    suspicious, reason = monitor.check_activity()
    return jsonify({"suspicious": suspicious, "reason": reason})

@app.route("/report")
def report():
    return render_template("report.html", username="Student123", exam_name="Math Exam", suspicious_logs=monitor.get_logs())

if __name__ == "__main__":
    app.run(debug=True)
