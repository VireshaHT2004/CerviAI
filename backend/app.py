# backend/app.py
import os, re, traceback
from io import BytesIO
from datetime import datetime
from pathlib import Path

from flask import Flask, request, redirect, url_for, render_template, send_file, flash, jsonify
from flask_cors import CORS
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load env vars from .env / .flaskenv if present
load_dotenv()

from .models import db, User, Image as ImageModel, Prediction, Report
from .utils import ensure_dirs, bgr_to_pil_rgb
# Use the shared prediction core
from .predict_folder import init_model, predict_bytes

# -------------------- Flask app --------------------
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-insecure-secret")
basedir = os.path.abspath(os.path.dirname(__file__))
# Database: prefer DATABASE_URL (e.g., Supabase Postgres) else fallback to local SQLite
db_url = os.environ.get("DATABASE_URL")
if db_url:
    # Supabase may give postgres://, switch to SQLAlchemy's psycopg2 dialect
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    # Make pooled connections more robust
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(basedir, 'data.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

ensure_dirs(["../database", "../models", "../static/js"])

with app.app_context():
    db.create_all()

# -------------------- Watch folders (raw + predictions) --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(basedir, ".."))
WATCH_INBOX_DIR = os.environ.get("WATCH_INBOX_DIR", os.path.join(PROJECT_ROOT, "data", "inbox"))
WATCH_OUT_DIR   = os.environ.get("WATCH_OUT_DIR",   os.path.join(PROJECT_ROOT, "data", "out"))
os.makedirs(WATCH_INBOX_DIR, exist_ok=True)
os.makedirs(WATCH_OUT_DIR, exist_ok=True)

def _safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        ext  = re.sub(r"[^A-Za-z0-9]+", "", ext)
        return f"{stem}.{ext}" if ext else stem
    else:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

# -------------------- YOLO settings (env) --------------------
MODEL_PATH = os.environ.get("MODEL_PATH", r"E:\cervix-app\models\best.pt")
YOLO_DEVICE = os.environ.get("YOLO_DEVICE")  # optional: "cpu" or "cuda"

YOLO_CONF   = float(os.environ.get("YOLO_CONF",  "0.10"))
YOLO_IOU    = float(os.environ.get("YOLO_IOU",   "0.50"))
YOLO_IMGSZ  = int(os.environ.get("YOLO_IMGSZ",  "896"))
YOLO_MAXDET = int(os.environ.get("YOLO_MAXDET", "300"))
MIN_AREA    = int(os.environ.get("MIN_AREA",    "30"))
DRAW_ON     = os.environ.get("DRAW_ON", "enhanced")
USE_PRE     = os.environ.get("USE_PREPROCESS", "1") not in ("0","false","False")
AUGMENT     = os.environ.get("YOLO_AUGMENT", "1") == "1"

# Load model once
try:
    init_model(MODEL_PATH, YOLO_DEVICE)
    yolo_error = None
except Exception as e:
    yolo_error = str(e)
    print("[YOLO] ERROR:", e)
    traceback.print_exc()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- Auth --------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("tech_dashboard" if current_user.role == "technician" else "doctor_dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "technician")
        if role not in ("technician", "doctor"):
            flash("Invalid role.", "danger"); return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger"); return render_template("register.html")
        user = User(name=name, email=email, password=generate_password_hash(password), role=role)
        db.session.add(user); db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f"Logged in as {user.role}", "success")
            return redirect(url_for("tech_dashboard" if user.role == "technician" else "doctor_dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# -------------------- Technician dashboard --------------------
@app.route("/dashboard/tech")
@login_required
def tech_dashboard():
    if current_user.role != "technician":
        flash("Access denied", "danger"); return redirect(url_for("login"))
    images = ImageModel.query.filter_by(uploaded_by=current_user.id).order_by(ImageModel.id.desc()).all()
    reports = Report.query.join(ImageModel).filter(ImageModel.uploaded_by == current_user.id).order_by(Report.id.desc()).all()
    return render_template("tech_dashboard.html", images=images, reports=reports, yolov7_loaded_err=yolo_error)

# -------------------- Doctor dashboard --------------------
@app.route("/dashboard/doctor")
@login_required
def doctor_dashboard():
    if current_user.role != "doctor":
        flash("Access denied", "danger"); return redirect(url_for("login"))
    pending = (ImageModel.query
               .join(Prediction, Prediction.image_id == ImageModel.id)
               .outerjoin(Report, Report.image_id == ImageModel.id)
               .filter(Report.id.is_(None))
               .order_by(ImageModel.id.desc())
               .all())
    return render_template("doctor_dashboard.html", images=pending)

@app.route("/review/<int:image_id>")
@login_required
def review(image_id):
    if current_user.role != "doctor":
        flash("Access denied", "danger"); return redirect(url_for("login"))
    img = ImageModel.query.get_or_404(image_id)
    pred = Prediction.query.filter_by(image_id=image_id).first()
    return render_template("review.html", image=img, prediction=pred)

@app.route("/report", methods=["POST"])
@login_required
def submit_report():
    if current_user.role != "doctor":
        flash("Access denied", "danger"); return redirect(url_for("login"))
    image_id = int(request.form["image_id"])
    status = request.form["status"]
    remarks = request.form.get("remarks", "")
    if Report.query.filter_by(image_id=image_id).first():
        flash("Report already exists.", "danger"); return redirect(url_for("doctor_dashboard"))
    rep = Report(image_id=image_id, doctor_id=current_user.id, status=status, remarks=remarks, created_at=datetime.utcnow())
    db.session.add(rep); db.session.commit()
    flash("Report submitted.", "success")
    return redirect(url_for("doctor_dashboard"))

# -------------------- Binary image endpoints --------------------
@app.route("/raw_image/<int:image_id>")
@login_required
def raw_image(image_id):
    img = ImageModel.query.get_or_404(image_id)
    ext = (img.filename.split(".")[-1].lower() if "." in img.filename else "png")
    mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ('tiff' if ext in ['tif','tiff'] else ext)}"
    return send_file(BytesIO(img.data), mimetype=mime, download_name=img.filename)

@app.route("/predicted_image/<int:image_id>")
@login_required
def predicted_image(image_id):
    p = Prediction.query.filter_by(image_id=image_id).first_or_404()
    fname = ImageModel.query.get(image_id).filename
    return send_file(BytesIO(p.annotated_image), mimetype="image/png", download_name=f"pred_{fname}.png")

# -------------------- PDF download --------------------
@app.route("/download/<int:report_id>")
@login_required
def download_report(report_id):
    report = Report.query.get_or_404(report_id)
    img = ImageModel.query.get(report.image_id)
    doctor = User.query.get(report.doctor_id)
    tech = User.query.get(img.uploaded_by)
    if current_user.role == "technician" and current_user.id != tech.id:
        return jsonify({"error": "forbidden"}), 403
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    text = c.beginText(40, 750); text.setFont("Helvetica", 12)
    text.textLine(f"Report ID: {report.id}")
    text.textLine(f"Image ID: {img.id} | Filename: {img.filename}")
    text.textLine(f"Uploaded by: {tech.name} <{tech.email}> at {img.upload_time.strftime('%Y-%m-%d %H:%M')}")
    text.textLine(f"Doctor: Dr. {doctor.name} <{doctor.email}> at {report.created_at.strftime('%Y-%m-%d %H:%M')}")
    text.textLine(f"Final Status: {report.status}")
    text.textLine(f"Remarks: {report.remarks or '—'}")
    c.drawText(text); c.showPage(); c.save(); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"report_{report.id}.pdf", mimetype="application/pdf")

# -------------------- Technician APIs --------------------
@app.route("/technician/api/upload", methods=["POST"])
@login_required
def tech_api_upload():
    if current_user.role != "technician":
        return jsonify({"success": False, "error": "forbidden"}), 403

    file = request.files.get("image")
    if not file:
        return jsonify({"success": False, "error": "missing file"}), 400
    data = file.read()
    if not data:
        return jsonify({"success": False, "error": "empty file"}), 400

    # Save raw upload to watch inbox
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    raw_name = f"{timestamp}_{_safe_name(file.filename)}"
    raw_path = os.path.join(WATCH_INBOX_DIR, raw_name)
    try:
        with open(raw_path, "wb") as f:
            f.write(data)
        print(f"[Upload] Saved raw to {raw_path}")
    except Exception as e:
        print("[Upload] Failed to save raw to disk:", e)

    # Store in DB
    img = ImageModel(filename=file.filename, data=data, uploaded_by=current_user.id, upload_time=datetime.utcnow())
    db.session.add(img); db.session.commit()

    # Predict via shared core (guarantee_one=True)
    try:
        res = predict_bytes(
            data,
            use_preprocess=USE_PRE,
            draw_on=DRAW_ON,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            max_det=YOLO_MAXDET,
            min_area=MIN_AREA,
            augment=AUGMENT,
            guarantee_one=True
        )
    except Exception as e:
        print("[Predict] ERROR:", e)
        return jsonify({"success": False, "error": "prediction failed"}), 500

    # Save annotated to DB
    total_cells = int(res["total_cells"])
    abnormal_cells = int(res["abnormal_cells"])
    status = "Positive" if total_cells > 0 else "Negative"
    confidence = float(res["confidence"])

    pred = Prediction(
        image_id=img.id,
        result=status,
        confidence=confidence,
        annotated_image=res["annotated_png"],
        created_at=datetime.utcnow(),
        total_cells=total_cells,
        abnormal_cells=abnormal_cells
    )
    db.session.add(pred); db.session.commit()

    # Save annotated PNG to watch out
    try:
        pred_name = f"pred_{Path(raw_name).stem}.png"
        out_path = os.path.join(WATCH_OUT_DIR, pred_name)
        with open(out_path, "wb") as f:
            f.write(res["annotated_png"])
        print(f"[Predict] Saved annotated to {out_path}")
    except Exception as e:
        print("[Predict] Failed to save annotated to disk:", e)

    return jsonify({
        "success": True,
        "image_id": img.id,
        "prediction": {
            "total_cells": total_cells,
            "abnormal_cells": abnormal_cells,
            "overall_confidence": confidence
        }
    }), 201

@app.route("/technician/api/stats")
@login_required
def tech_api_stats():
    if current_user.role != "technician":
        return jsonify({"error": "forbidden"}), 403
    total_uploads = ImageModel.query.filter_by(uploaded_by=current_user.id).count()
    pending_review = (ImageModel.query
                      .filter_by(uploaded_by=current_user.id)
                      .join(Prediction, Prediction.image_id == ImageModel.id, isouter=False)
                      .outerjoin(Report, Report.image_id == ImageModel.id)
                      .filter(Report.id.is_(None))
                      .count())
    completed = (Report.query
                 .join(ImageModel, ImageModel.id == Report.image_id)
                 .filter(ImageModel.uploaded_by == current_user.id)
                 .count())
    return jsonify({"total_uploads": total_uploads, "pending_review": pending_review, "completed": completed})

@app.route("/technician/api/images")
@login_required
def tech_api_images():
    if current_user.role != "technician": return jsonify({"error": "forbidden"}), 403
    imgs = (ImageModel.query.filter_by(uploaded_by=current_user.id).order_by(ImageModel.id.desc()).all())
    out = []
    for img in imgs:
        pred, rep = img.prediction, img.report
        status = "pending" if not pred else ("predicted" if not rep else "reviewed")
        out.append({
            "id": img.id,
            "filename": img.filename,
            "upload_time": img.upload_time.isoformat(),
            "status": status,
            "prediction": None if not pred else {
                "cell_count": pred.total_cells or 0,
                "abnormal_count": pred.abnormal_cells or 0,
                "confidence": float(pred.confidence)
            },
            "report": None if not rep else {
                "id": rep.id, "status": rep.status, "created_at": rep.created_at.isoformat()
            }
        })
    return jsonify({"images": out})

@app.route("/technician/api/reports")
@login_required
def tech_api_reports():
    if current_user.role != "technician": return jsonify({"error": "forbidden"}), 403
    reps = (Report.query
            .join(ImageModel, ImageModel.id == Report.image_id)
            .join(User, User.id == Report.doctor_id)
            .filter(ImageModel.uploaded_by == current_user.id)
            .order_by(Report.id.desc()).all())
    out = []
    for r in reps:
        img = ImageModel.query.get(r.image_id); doc = User.query.get(r.doctor_id)
        out.append({
            "id": r.id, "image_id": r.image_id, "filename": img.filename if img else "",
            "status": r.status, "doctor": doc.name if doc else "", "created_at": r.created_at.isoformat()
        })
    return jsonify({"reports": out})

@app.route("/technician/api/download/report/<int:report_id>")
@login_required
def tech_api_download_report(report_id):
    return download_report(report_id)

# -------------------- Legacy API (optional manual upload) --------------------
@app.route("/upload", methods=["POST"])
@login_required
def api_upload():
    if current_user.role != "technician": return jsonify({"error": "forbidden"}), 403
    file = request.files.get("image")
    if not file: return jsonify({"error": "missing file"}), 400
    data = file.read()
    if not data: return jsonify({"error": "empty file"}), 400

    # mirror to inbox
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    raw_name = f"{timestamp}_{_safe_name(file.filename)}"
    raw_path = os.path.join(WATCH_INBOX_DIR, raw_name)
    try:
        with open(raw_path, "wb") as f:
            f.write(data)
    except Exception as e:
        print("[Upload] Failed to save raw to disk:", e)

    img = ImageModel(filename=file.filename, data=data, uploaded_by=current_user.id, upload_time=datetime.utcnow())
    db.session.add(img); db.session.commit()

    # predict
    res = predict_bytes(
        data,
        use_preprocess=USE_PRE,
        draw_on=DRAW_ON,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAXDET,
        min_area=MIN_AREA,
        augment=AUGMENT,
        guarantee_one=True
    )

    # write pred to db + disk
    pred = Prediction(
        image_id=img.id,
        result=("Positive" if res["total_cells"] > 0 else "Negative"),
        confidence=float(res["confidence"]),
        annotated_image=res["annotated_png"],
        created_at=datetime.utcnow(),
        total_cells=int(res["total_cells"]),
        abnormal_cells=int(res["abnormal_cells"]),
    )
    db.session.add(pred); db.session.commit()

    try:
        out_path = os.path.join(WATCH_OUT_DIR, f"pred_{Path(raw_name).stem}.png")
        with open(out_path, "wb") as f:
            f.write(res["annotated_png"])
    except Exception as e:
        print("[Predict] Failed to save annotated to disk:", e)

    return jsonify({"image_id": img.id, "predicted": True}), 201

# -------------------- Health --------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "yolo_loaded": yolo_error is None, "yolo_error": yolo_error})

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    os.makedirs(WATCH_INBOX_DIR, exist_ok=True)
    os.makedirs(WATCH_OUT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
