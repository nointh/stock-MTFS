FROM python:3.10-slim

COPY ./src /app

WORKDIR /app

COPY requirements.txt .

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]