FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7861

ENV GRADIO_SERVER_NAME="0.0.0.0"

ENV GRADIO_SERVER_PORT="7861"

CMD ["python", "gradio_app.py"]