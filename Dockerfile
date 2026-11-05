FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('VerMichel/german-reviews-distilbert'); AutoModelForSequenceClassification.from_pretrained('VerMichel/german-reviews-distilbert')"

COPY api/ ./api/
COPY core/ ./core/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
