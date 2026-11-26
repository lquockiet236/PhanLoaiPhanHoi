import os
from datetime import datetime
from functools import wraps
import sqlite3
from pathlib import Path

from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from test import LABEL_MAPPING, predict_topic

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "feedback.db"
STATUSES = ("Mới", "Đang xử lý", "Đã phản hồi", "Đã đóng")
CATEGORIES = tuple(LABEL_MAPPING.values())
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "campus-voice-development-key")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            category TEXT NOT NULL,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Mới',
            response TEXT,
            responded_at TEXT,
            user_id INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    columns = {row["name"] for row in db.execute("PRAGMA table_info(feedback)").fetchall()}
    if "priority" in columns:
        db.execute("ALTER TABLE feedback DROP COLUMN priority")

    for column, definition in {
        "response": "TEXT",
        "responded_at": "TEXT",
        "user_id": "INTEGER",
        "category_confidence": "REAL",
    }.items():
        if column not in columns:
            db.execute(f"ALTER TABLE feedback ADD COLUMN {column} {definition}")

    db.execute(
        """CREATE TABLE IF NOT EXISTS feedback_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL REFERENCES feedback(id) ON DELETE CASCADE,
            author_id INTEGER,
            author_name TEXT NOT NULL,
            author_role TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    db.execute("UPDATE feedback SET status = 'Đã đóng' WHERE status = 'Đã phản hồi'")

    admin = db.execute("SELECT id FROM users WHERE email = ?", ("admin@campus.local",)).fetchone()
    if admin is None:
        db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("Quản trị viên", "admin@campus.local", generate_password_hash("admin123"), "admin", datetime.now().isoformat(timespec="minutes")),
        )
    db.commit()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if current_user()["role"] != "admin":
            flash("Bạn không có quyền truy cập khu vực quản trị.", "error")
            return redirect(url_for("index"))
        return view(*args, **kwargs)
    return wrapped_view


@app.context_processor
def inject_globals():
    db = get_db()
    return {
        "current_user": current_user(),
        "total_feedback": db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
        "pending_feedback": db.execute("SELECT COUNT(*) FROM feedback WHERE status IN ('Mới', 'Đang xử lý')").fetchone()[0],
        "feedback_categories": CATEGORIES,
    }


def feedback_threads(rows):
    feedback = [dict(row) for row in rows]
    if not feedback:
        return feedback
    ids = [item["id"] for item in feedback]
    placeholders = ",".join("?" for _ in ids)
    messages = get_db().execute(
        f"SELECT * FROM feedback_messages WHERE feedback_id IN ({placeholders}) ORDER BY id", ids
    ).fetchall()
    grouped = {feedback_id: [] for feedback_id in ids}
    for message in messages:
        grouped[message["feedback_id"]].append(dict(message))
    for item in feedback:
        item["messages"] = grouped[item["id"]]
    return feedback


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("admin" if current_user()["role"] == "admin" else "index"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not name or not email or len(password) < 6:
            flash("Vui lòng nhập đủ thông tin và mật khẩu tối thiểu 6 ký tự.", "error")
        elif password != confirm_password:
            flash("Mật khẩu xác nhận không khớp.", "error")
        else:
            try:
                db = get_db()
                cursor = db.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (name, email, generate_password_hash(password), datetime.now().isoformat(timespec="minutes")),
                )
                db.commit()
                session.clear()
                session["user_id"] = cursor.lastrowid
                flash("Đăng ký thành công. Chào mừng bạn đến với Campus Voice!", "success")
                return redirect(url_for("index"))
            except sqlite3.IntegrityError:
                flash("Email này đã được đăng ký.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    remembered_email = request.cookies.get("remembered_email", "")
    if current_user():
        return redirect(url_for("admin" if current_user()["role"] == "admin" else "index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Email hoặc mật khẩu không đúng.", "error")
            return render_template("login.html", remembered_email=email if request.form.get("remember_me") else remembered_email)

        session.clear()
        session["user_id"] = user["id"]
        redirect_response = redirect(url_for("admin" if user["role"] == "admin" else "index"))
        if request.form.get("remember_me"):
            redirect_response.set_cookie("remembered_email", email, max_age=60 * 60 * 24 * 30, httponly=True, samesite="Lax")
        else:
            redirect_response.delete_cookie("remembered_email")
        return redirect_response
    return render_template("login.html", remembered_email=remembered_email)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    user = current_user()
    if current_user()["role"] == "admin":
        return redirect(url_for("admin"))

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        if not subject or not message:
            flash("Vui lòng nhập tiêu đề và nội dung phản hồi.", "error")
        else:
            try:
                _, predicted_category, confidence = predict_topic(f"{subject}. {message}")
                category = (
                    predicted_category
                    if confidence >= CLASSIFICATION_CONFIDENCE_THRESHOLD
                    else LABEL_MAPPING[3]
                )
            except Exception:
                app.logger.exception("Unable to classify feedback")
                flash("Model phân loại chưa sẵn sàng. Vui lòng kiểm tra dữ liệu và thử lại.", "error")
                return redirect(url_for("index"))
            db = get_db()
            cursor = db.execute(
                """INSERT INTO feedback
                   (name, email, category, subject, message, status, user_id, created_at, category_confidence)
                   VALUES (?, ?, ?, ?, ?, 'Mới', ?, ?, ?)""",
                (
                    user["name"], user["email"], category, subject, message, user["id"],
                    datetime.now().isoformat(timespec="minutes"),
                    confidence,
                ),
            )
            db.commit()
            return redirect(url_for("index", _anchor=f"feedback-{cursor.lastrowid}"))

    rows = get_db().execute(
        """SELECT feedback.*,
                  (SELECT MAX(created_at) FROM feedback_messages WHERE feedback_id = feedback.id) AS last_message_at,
                  (SELECT MAX(id) FROM feedback_messages WHERE feedback_id = feedback.id) AS last_message_id
           FROM feedback
           WHERE user_id = ?
           ORDER BY COALESCE(last_message_at, created_at) DESC, last_message_id DESC, id DESC""",
        (user["id"],),
    ).fetchall()
    return render_template("student_dashboard.html", feedback=feedback_threads(rows), statuses=STATUSES)


@app.get("/api/my-feedback-stats")
@login_required
def my_feedback_stats():
    user = current_user()
    if user["role"] != "user":
        return jsonify({"error": "forbidden"}), 403
    row = get_db().execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN status = 'Mới' THEN 1 ELSE 0 END) AS pending,
                  SUM(CASE WHEN status IN ('Đã phản hồi', 'Đã đóng') THEN 1 ELSE 0 END) AS responded
           FROM feedback WHERE user_id = ?""",
        (user["id"],),
    ).fetchone()
    return jsonify({key: int(row[key] or 0) for key in ("total", "pending", "responded")})


@app.route("/admin")
@admin_required
def admin():
    search = request.args.get("q", "").strip()
    category = request.args.get("category", "Tất cả")
    current_year = datetime.now().year
    try:
        chart_year = int(request.args.get("year", current_year))
    except ValueError:
        chart_year = current_year
    if chart_year < 2000 or chart_year > current_year + 1:
        chart_year = current_year
    query = "SELECT * FROM feedback WHERE 1=1"
    params = []

    if search:
        query += " AND (name LIKE ? OR email LIKE ? OR subject LIKE ? OR message LIKE ?)"
        pattern = f"%{search}%"
        params.extend([pattern] * 4)
    if category in CATEGORIES:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY id DESC"

    feedback = feedback_threads(get_db().execute(query, params).fetchall())
    year_rows = get_db().execute(
        "SELECT strftime('%m', created_at) AS month, category, COUNT(*) AS count "
        "FROM feedback WHERE strftime('%Y', created_at) = ? GROUP BY month, category",
        (str(chart_year),),
    ).fetchall()
    counts_by_month = {(row["month"], row["category"]): row["count"] for row in year_rows}
    max_month_total = 0
    monthly_chart = []
    for month_number in range(1, 13):
        month_key = f"{month_number:02d}"
        category_counts = [counts_by_month.get((month_key, name), 0) for name in CATEGORIES]
        month_total = sum(category_counts)
        max_month_total = max(max_month_total, month_total)
        monthly_chart.append({"label": f"T{month_number}", "counts": category_counts, "total": month_total})
    total_feedback = get_db().execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    year_total = sum(month["total"] for month in monthly_chart)
    years = {current_year, chart_year}
    years.update(
        int(row["year"])
        for row in get_db().execute(
            "SELECT DISTINCT strftime('%Y', created_at) AS year FROM feedback "
            "WHERE strftime('%Y', created_at) IS NOT NULL"
        ).fetchall()
    )
    return render_template(
        "admin_dashboard.html", feedback=feedback, search=search, category=category,
        monthly_chart=monthly_chart, chart_year=chart_year, chart_years=sorted(years, reverse=True),
        max_month_total=max_month_total, total_feedback=total_feedback, year_total=year_total,
        active_tab="feedback" if request.args.get("view") == "feedback" else None,
    )


@app.post("/admin/feedback/<int:feedback_id>/respond")
@admin_required
def respond_feedback(feedback_id):
    response = request.form.get("body", "").strip()
    feedback = get_db().execute("SELECT id, status FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
    if feedback is None:
        flash("Không tìm thấy phản hồi.", "error")
    elif feedback["status"] == "Đã đóng":
        flash("Ý kiến này đã đóng, không thể gửi thêm tin nhắn.", "error")
    elif not response:
        flash("Nội dung trả lời không được để trống.", "error")
    else:
        user = current_user()
        now = datetime.now().isoformat(timespec="minutes")
        db = get_db()
        db.execute(
            "INSERT INTO feedback_messages (feedback_id, author_id, author_name, author_role, body, created_at) VALUES (?, ?, ?, 'admin', ?, ?)",
            (feedback_id, user["id"], user["name"], response, now),
        )
        db.execute(
            "UPDATE feedback SET response = ?, responded_at = ?, status = 'Đã đóng' WHERE id = ?",
            (response, now, feedback_id),
        )
        db.commit()
    return redirect(request.referrer or url_for("admin"))


with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5001")), debug=True)
