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
