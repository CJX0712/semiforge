FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY semiforge ./semiforge
COPY examples ./examples
COPY tests ./tests
COPY pyproject.toml README.md ./

CMD ["python", "examples/run_demo.py"]
