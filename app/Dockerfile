# syntax=docker/dockerfile:1

FROM python:3.12.13-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV EMULATE=0

EXPOSE 5001
CMD ["gunicorn","--workers", "2","--timeout","120","--preload", "--bind", "0.0.0.0:5001","phagegap:app"]
