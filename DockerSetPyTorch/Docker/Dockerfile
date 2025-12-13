FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu


COPY main.py .

EXPOSE 5000
CMD ["python", "-m", "uvicorn", "main:app", "--host","0.0.0.0","--port","5000"]
