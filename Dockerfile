FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir '.[runpod,excel]'
COPY handler.py /app/handler.py
COPY config /app/config
COPY scripts /app/scripts
RUN mkdir -p /app/results /app/local && chown -R 10001:10001 /app/results /app/local
USER 10001:10001
CMD ["python", "-u", "handler.py"]
