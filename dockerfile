FROM python:3.11-slim

ENV PYTHONBUFFERED=1

RUN pip install pandas mlxtend

COPY . .

CMD ["python", "main.py"]