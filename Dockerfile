FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY run.sh solution.py /app/
COPY src /app/src
COPY models /app/models
COPY tessdata_best /app/tessdata_best
RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]
