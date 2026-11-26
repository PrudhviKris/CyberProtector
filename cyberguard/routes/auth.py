import re

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from ..models import User, db

bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.verify"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        errors = []
        if not re.fullmatch(r"[A-Za-z0-9_]{3,40}", username):
            errors.append("Username must be 3-40 letters, digits or underscores.")
        if not EMAIL_RE.match(email):
            errors.append("Enter a valid email address.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if User.query.filter((User.username == username) | (User.email == email)).first():
            errors.append("That username or email is already registered.")
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth/register.html", username=username, email=email), 400
        # Public sign-ups are ALWAYS regular users. Admins can only be created by the owner,
        # via the ADMIN_* environment variables or the `flask create-admin` command.
        user = User(username=username, email=email, is_admin=False)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Welcome to CyberGuard!", "success")
        return redirect(url_for("main.verify"))
    return render_template("auth/register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.verify"))
    if request.method == "POST":
        ident = request.form.get("identifier", "").strip()
        user = User.query.filter((User.username == ident) | (User.email == ident.lower())).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user, remember=bool(request.form.get("remember")))
            return redirect(_safe_next(request.args.get("next")) or url_for("main.verify"))
        flash("Invalid username/email or password.", "error")
        return render_template("auth/login.html", identifier=ident), 401
    return render_template("auth/login.html")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.home"))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    if request.method == "POST" and request.form.get("action") == "rotate":
        current_user.rotate_api_key()
        db.session.commit()
        flash("A new API key has been generated.", "success")
        return redirect(url_for("auth.account"))
    return render_template("auth/account.html")
