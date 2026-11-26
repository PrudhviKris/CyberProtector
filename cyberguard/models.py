import secrets
from datetime import datetime, timezone

from flask_login import LoginManager, UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    api_key = db.Column(db.String(64), unique=True, index=True, default=lambda: secrets.token_hex(24))
    created_at = db.Column(db.DateTime, default=utcnow)
    analyses = db.relationship("AnalysisRecord", backref="user", lazy="dynamic")

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    def rotate_api_key(self):
        self.api_key = secrets.token_hex(24)


class AnalysisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(20), nullable=False, index=True)
    score = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(10), nullable=False)
    top_category = db.Column(db.String(30))
    categories = db.Column(db.JSON, default=dict)
    source = db.Column(db.String(10), default="web")    # web | api | batch
    created_at = db.Column(db.DateTime, default=utcnow, index=True)

    @classmethod
    def from_analysis(cls, a, user=None, source="web"):
        return cls(user_id=user.id if user else None, text=a.text, label=a.label, score=a.score,
                   severity=a.severity, top_category=a.top_category, categories=a.categories, source=source)


class Ticket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(12), default="open")    # open | resolved
    created_at = db.Column(db.DateTime, default=utcnow)


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))
