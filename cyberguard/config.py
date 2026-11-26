import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'instance' / 'cyberguard.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024          # 5 MB uploads (batch CSV)
    BASELINE_MODEL = os.environ.get("BASELINE_MODEL", str(ROOT / "artifacts" / "baseline.joblib"))
    # set TRANSFORMER_MODEL="" to run with the classical model only
    TRANSFORMER_MODEL = os.environ.get("TRANSFORMER_MODEL", "unitary/toxic-bert")
    THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
    BATCH_LIMIT = int(os.environ.get("BATCH_LIMIT", "500"))
    ARTIFACTS_DIR = ROOT / "artifacts"


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    TRANSFORMER_MODEL = ""
