"""REST API: POST /api/v1/analyze  (header X-API-Key: <key from /account>)"""
from flask import Blueprint, current_app, jsonify, request

from .. import get_predictor
from ..models import AnalysisRecord, User, db

bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _auth():
    key = request.headers.get("X-API-Key") or request.args.get("api_key")
    return User.query.filter_by(api_key=key).first() if key else None


@bp.get("/health")
def health():
    p = get_predictor()
    return jsonify(status="ok", baseline=True, transformer=p.transformer_ready,
                   transformer_model=p.transformer_name)


@bp.post("/analyze")
def analyze():
    user = _auth()
    if user is None:
        return jsonify(error="Missing or invalid API key. Send it in the X-API-Key header."), 401
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    single = texts is None
    if single:
        texts = [data.get("text")]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        return jsonify(error="Provide 'text' (string) or 'texts' (non-empty list of strings)."), 400
    if len(texts) > current_app.config["BATCH_LIMIT"]:
        return jsonify(error=f"At most {current_app.config['BATCH_LIMIT']} texts per request."), 413
    explain = bool(data.get("explain", single))
    results = get_predictor().analyze_many(texts, explain=explain)
    db.session.add_all(AnalysisRecord.from_analysis(r, user, "api") for r in results)
    db.session.commit()
    payload = [r.to_dict() for r in results]
    return jsonify(payload[0] if single else {"results": payload})
