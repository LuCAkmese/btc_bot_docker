FROM python:3.10.13-slim

RUN apt-get update && apt-get install -y build-essential

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

EXPOSE 8050
CMD ["python", "wt_bot_fixed.py"]
