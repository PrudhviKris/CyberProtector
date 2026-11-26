import logging
import os
import threading

import click
from flask import Flask, current_app
from flask_wtf.csrf import CSRFProtect

from .config import ROOT, Config
from .models import User, db, login_manager

csrf = CSRFProtect()


def get_predictor():
    return current_app.extensions["predictor"]


def _bootstrap_admin(app):
    """Create/promote the owner's admin account from environment variables, if they are set."""
    username, email, password = (app.config.get(k, "") for k in ("ADMIN_USERNAME", "ADMIN_EMAIL", "ADMIN_PASSWORD"))
    if not (username and email and password):
        return
    if len(password) < 12:
        app.logger.error("ADMIN_PASSWORD must be at least 12 characters; admin account not created.")
        return
    user = User.query.filter((User.username == username) | (User.email == email.lower())).first()
    if user is None:
        user = User(username=username, email=email.lower())
        db.session.add(user)
    else:  # make sure the account fully belongs to the owner (e.g. if someone pre-registered the name)
        user.username, user.email = username, email.lower()
        user.rotate_api_key()
    user.set_password(password)   # env var is the source of truth for the owner's password
    user.is_admin = True
    db.session.commit()
    app.logger.info("Admin account '%s' is ready.", username)


def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)
    (ROOT / "instance").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    from .ml.predictor import Predictor
    predictor = Predictor(app.config["BASELINE_MODEL"], app.config["TRANSFORMER_MODEL"] or None,
                          threshold=app.config["THRESHOLD"])
    app.extensions["predictor"] = predictor
    if predictor.transformer_name and not app.testing and os.environ.get("WARMUP", "1") == "1":
        threading.Thread(target=predictor.warmup, daemon=True).start()

    from .routes import admin, api, auth, main
    app.register_blueprint(main.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(api.bp)
    csrf.exempt(api.bp)

    from .ml.predictor import CATEGORY_LABELS

    @app.context_processor
    def inject_globals():
        return {"CATEGORY_LABELS": CATEGORY_LABELS}

    with app.app_context():
        db.create_all()
        _bootstrap_admin(app)

    # if not app.debug and not app.testing and app.config["SECRET_KEY"] == "dev-change-me":
    #     app.logger.warning("SECRET_KEY is the default value. Set a strong SECRET_KEY before deploying!")
    if not app.debug and not app.testing and app.config["SECRET_KEY"] in ("dev-change-me", "change-me-in-production"):
        raise RuntimeError("SECRET_KEY is not set. Refusing to start with a public default key.")

    @app.cli.command("create-admin")
    @click.option("--username", prompt=True)
    @click.option("--email", prompt=True)
    @click.password_option()
    def create_admin(username, email, password):
        """Create (or promote) an administrator account."""
        user = User.query.filter((User.username == username) | (User.email == email)).first()
        if user is None:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
        user.is_admin = True
        db.session.commit()
        click.echo(f"Admin ready: {user.username}")

    return app
