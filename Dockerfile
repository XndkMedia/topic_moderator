FROM python:3.10-bullseye
WORKDIR /app

COPY . .
RUN pip install poetry
RUN poetry install

CMD poetry run uvicorn ai_server.main:app --host 0.0.0.0 --port 8000