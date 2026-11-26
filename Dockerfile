FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf PORT=7860

# Hugging Face runs the container as user 1000, not root
RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# let user 1000 create instance/ (SQLite) and the model cache
RUN mkdir -p instance .hf && chown -R user:user /app
USER user

# download toxic-bert during the build so startup doesn't have to
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; AutoTokenizer.from_pretrained('unitary/toxic-bert'); M.from_pretrained('unitary/toxic-bert')"

EXPOSE 7860
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -b 0.0.0.0:${PORT} --timeout 120 run:app"]
