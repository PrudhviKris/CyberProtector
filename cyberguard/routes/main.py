import csv
import io
import json

from flask import (Blueprint, Response, abort, current_app, flash, redirect, render_template,
                   request, send_from_directory, session, url_for)
from flask_login import current_user, login_required

from .. import get_predictor
from ..ml.predictor import highlight_spans
from ..models import AnalysisRecord, Ticket, db

bp = Blueprint("main", __name__)
TEXT_COLUMNS = ("text", "comment", "message", "headline", "content", "tweet")


@bp.route("/")
def home():
    return render_template("home.html")


@bp.route("/verify", methods=["GET", "POST"])
def verify():
    result = spans = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Please enter some text to analyse.", "error")
        else:
            result = get_predictor().analyze(text)
            spans = highlight_spans(result.text, result.highlights)
            db.session.add(AnalysisRecord.from_analysis(
                result, current_user if current_user.is_authenticated else None, "web"))
            db.session.commit()
    return render_template("verify.html", result=result, spans=spans, text=text)


@bp.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    flt = request.args.get("filter", "all")
    q = current_user.analyses.order_by(AnalysisRecord.created_at.desc())
    if flt in ("bullying", "non-bullying"):
        q = q.filter_by(label=flt)
    pagination = q.paginate(page=page, per_page=15, error_out=False)
    return render_template("history.html", pagination=pagination, flt=flt)


@bp.route("/history/export.csv")
@login_required
def history_export():
    rows = current_user.analyses.order_by(AnalysisRecord.created_at.desc()).all()
    return _csv_response("my_history.csv", ["created_at", "source", "label", "score", "severity", "top_category", "text"],
                         ([r.created_at.isoformat(timespec="seconds"), r.source, r.label, f"{r.score:.4f}",
                           r.severity, r.top_category or "", r.text] for r in rows))


@bp.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith((".csv", ".txt")):
            flash("Upload a .csv (with a 'text' column) or a .txt file (one message per line).", "error")
            return redirect(url_for("main.batch"))
        raw = f.read().decode("utf-8-sig", errors="replace")
        if f.filename.lower().endswith(".txt"):
            texts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        else:
            reader = csv.DictReader(io.StringIO(raw))
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            col = next((cols[c] for c in TEXT_COLUMNS if c in cols), None)
            if col is None:
                flash(f"CSV needs one of these columns: {', '.join(TEXT_COLUMNS)}.", "error")
                return redirect(url_for("main.batch"))
            texts = [(row.get(col) or "").strip() for row in reader]
            texts = [t for t in texts if t]
        limit = current_app.config["BATCH_LIMIT"]
        if not texts:
            flash("No messages found in the file.", "error")
            return redirect(url_for("main.batch"))
        if len(texts) > limit:
            flash(f"Only the first {limit} messages were analysed.", "info")
            texts = texts[:limit]
        results = get_predictor().analyze_many(texts, explain=False)
        records = [AnalysisRecord.from_analysis(r, current_user, "batch") for r in results]
        db.session.add_all(records)
        db.session.commit()
        session["last_batch"] = [r.id for r in records]
        flagged = sum(r.is_bullying for r in results)
        return render_template("batch.html", results=results, flagged=flagged)
    return render_template("batch.html", results=None)


@bp.route("/batch/last.csv")
@login_required
def batch_export():
    ids = session.get("last_batch") or []
    rows = AnalysisRecord.query.filter(AnalysisRecord.id.in_(ids), AnalysisRecord.user_id == current_user.id).all()
    if not rows:
        abort(404)
    return _csv_response("batch_results.csv", ["text", "label", "score", "severity", "top_category"],
                         ([r.text, r.label, f"{r.score:.4f}", r.severity, r.top_category or ""] for r in rows))


@bp.route("/helpdesk", methods=["GET", "POST"])
def helpdesk():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()[:120] or "General query"
        message = request.form.get("message", "").strip()
        if "@" not in email or not message:
            flash("Please provide a valid email and a message.", "error")
            return render_template("helpdesk.html", email=email, subject=subject, message=message), 400
        db.session.add(Ticket(email=email, subject=subject, message=message,
                              user_id=current_user.id if current_user.is_authenticated else None))
        db.session.commit()
        flash("Thanks for reaching out. Your ticket has been recorded and our team will reply by email.", "success")
        return redirect(url_for("main.helpdesk"))
    email = current_user.email if current_user.is_authenticated else ""
    return render_template("helpdesk.html", email=email)


@bp.route("/about")
def about():
    return render_template("about.html")


@bp.route("/model")
def model_card():
    path = current_app.config["ARTIFACTS_DIR"] / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else None
    p = get_predictor()
    return render_template("model.html", metrics=metrics, transformer=p.transformer_name,
                           transformer_ready=p.transformer_ready)


@bp.route("/artifacts/<path:name>")
def artifact(name):
    if not name.endswith(".png"):
        abort(404)
    return send_from_directory(current_app.config["ARTIFACTS_DIR"], name)


@bp.route("/api-docs")
def api_docs():
    return render_template("api_docs.html")


def _csv_response(filename, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})
