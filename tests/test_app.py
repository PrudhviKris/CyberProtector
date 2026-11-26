import io

import pytest

from cyberguard import create_app
from cyberguard.config import TestConfig
from cyberguard.ml.text import clean_text
from cyberguard.models import AnalysisRecord, Ticket, User, db


@pytest.fixture()
def app():
    app = create_app(TestConfig)
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def register(client, username="alice", email="alice@example.com", password="password123"):
    return client.post("/register", data={"username": username, "email": email, "password": password},
                       follow_redirects=True)


# ---------- preprocessing / model ----------
def test_clean_text_decodes_obfuscation():
    assert clean_text("You are $tupid @bob http://x.co F*CK") == "you are stupid fuck"
    assert clean_text("sooooo baaaad") == "soo baad"
    assert clean_text(None) == ""


def test_predictor_separates_obvious_cases(app):
    p = app.extensions["predictor"]
    bad = p.analyze("you are a stupid worthless idiot, go kill yourself")
    good = p.analyze("thanks for fixing the references on this article, great work")
    assert bad.is_bullying and bad.score > 0.5
    assert not good.is_bullying and good.score < 0.5
    assert any(h["token"] in ("stupid", "idiot", "worthless", "kill") for h in bad.highlights if h["weight"] > 0)


# ---------- pages ----------
@pytest.mark.parametrize("path", ["/", "/verify", "/about", "/helpdesk", "/model", "/api-docs", "/login", "/register",
                                  "/api/v1/health"])
def test_public_pages(client, path):
    assert client.get(path).status_code == 200


def test_verify_records_analysis(client, app):
    r = client.post("/verify", data={"text": "you are an idiot"})
    assert r.status_code == 200 and b"Bullying detected" in r.data
    with app.app_context():
        assert AnalysisRecord.query.count() == 1


def test_login_required_pages_redirect(client):
    for path in ("/history", "/batch", "/account", "/admin/"):
        assert client.get(path).status_code == 302


# ---------- auth / roles ----------
def test_first_user_is_admin_second_is_not(client, app):
    register(client)
    client.post("/logout")
    register(client, "bob", "bob@example.com")
    with app.app_context():
        assert User.query.filter_by(username="alice").one().is_admin
        assert not User.query.filter_by(username="bob").one().is_admin
    assert client.get("/admin/").status_code == 403


def test_login_and_bad_password(client):
    register(client)
    client.post("/logout")
    assert client.post("/login", data={"identifier": "alice", "password": "wrong"}).status_code == 401
    r = client.post("/login", data={"identifier": "alice@example.com", "password": "password123"})
    assert r.status_code == 302


def test_admin_dashboard_and_history(client):
    register(client)
    client.post("/verify", data={"text": "you are a moron"})
    assert b"moron" in client.get("/history").data
    assert client.get("/admin/").status_code == 200
    assert client.get("/history/export.csv").mimetype == "text/csv"


def test_batch_csv(client):
    register(client)
    data = {"file": (io.BytesIO(b"comment\nhave a lovely day\nshut up you idiot\n"), "msgs.csv")}
    r = client.post("/batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Messages analysed" in r.data
    out = client.get("/batch/last.csv").data.decode()
    assert "idiot" in out and out.count("\n") == 3


def test_helpdesk_ticket(client, app):
    client.post("/helpdesk", data={"email": "x@y.com", "subject": "Wrong prediction", "message": "hi"})
    with app.app_context():
        assert Ticket.query.one().subject == "Wrong prediction"


# ---------- API ----------
def test_api_requires_key(client):
    assert client.post("/api/v1/analyze", json={"text": "hi"}).status_code == 401


def test_api_single_and_batch(client, app):
    register(client)
    with app.app_context():
        key = User.query.one().api_key
    h = {"X-API-Key": key}
    r = client.post("/api/v1/analyze", json={"text": "you are pathetic and ugly"}, headers=h)
    assert r.status_code == 200 and r.json["label"] == "bullying" and "highlights" in r.json
    r = client.post("/api/v1/analyze", json={"texts": ["hello friend", "die you idiot"]}, headers=h)
    labels = [x["label"] for x in r.json["results"]]
    assert labels == ["non-bullying", "bullying"]
    assert client.post("/api/v1/analyze", json={"texts": []}, headers=h).status_code == 400
