from collections import Counter
from datetime import timedelta
from functools import wraps

from flask import Blueprint, abort, flash, redirect, render_template, url_for
from flask_login import current_user, login_required
from sqlalchemy import func

from ..models import AnalysisRecord, Ticket, User, db, utcnow

bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*a, **kw):
        if not current_user.is_admin:
            abort(403)
        return fn(*a, **kw)
    return wrapper


@bp.route("/")
@admin_required
def dashboard():
    total = AnalysisRecord.query.count()
    flagged = AnalysisRecord.query.filter_by(label="bullying").count()
    since = utcnow() - timedelta(days=13)
    day = func.date(AnalysisRecord.created_at)
    daily_rows = (db.session.query(day, AnalysisRecord.label, func.count())
                  .filter(AnalysisRecord.created_at >= since).group_by(day, AnalysisRecord.label).all())
    days = [(since + timedelta(days=i)).date().isoformat() for i in range(14)]
    daily = {d: {"bullying": 0, "non-bullying": 0} for d in days}
    for d, label, n in daily_rows:
        if str(d) in daily:
            daily[str(d)][label] = n
    cats = Counter(c for (c,) in db.session.query(AnalysisRecord.top_category)
                   .filter(AnalysisRecord.top_category.isnot(None)).all())
    sev = dict(db.session.query(AnalysisRecord.severity, func.count())
               .filter(AnalysisRecord.label == "bullying").group_by(AnalysisRecord.severity).all())
    sources = dict(db.session.query(AnalysisRecord.source, func.count()).group_by(AnalysisRecord.source).all())
    recent = (AnalysisRecord.query.filter_by(label="bullying")
              .order_by(AnalysisRecord.created_at.desc()).limit(10).all())
    tickets = Ticket.query.order_by(Ticket.status.asc(), Ticket.created_at.desc()).limit(25).all()
    return render_template(
        "admin/dashboard.html", total=total, flagged=flagged, users=User.query.count(),
        open_tickets=Ticket.query.filter_by(status="open").count(),
        chart={"days": days, "bullying": [daily[d]["bullying"] for d in days],
               "clean": [daily[d]["non-bullying"] for d in days],
               "categories": dict(cats.most_common()), "severity": sev, "sources": sources},
        recent=recent, tickets=tickets)


@bp.post("/tickets/<int:tid>/toggle")
@admin_required
def toggle_ticket(tid):
    t = db.get_or_404(Ticket, tid)
    t.status = "resolved" if t.status == "open" else "open"
    db.session.commit()
    flash(f"Ticket #{t.id} marked {t.status}.", "success")
    return redirect(url_for("admin.dashboard") + "#tickets")


@bp.route("/users")
@admin_required
def users():
    rows = (db.session.query(User, func.count(AnalysisRecord.id))
            .outerjoin(AnalysisRecord).group_by(User.id).order_by(User.created_at.desc()).all())
    return render_template("admin/users.html", rows=rows)
