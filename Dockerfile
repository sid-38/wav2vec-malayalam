# syntax=docker/dockerfile:1
FROM python:3-slim-bullseye
WORKDIR /code
RUN apt-get update && apt-get install -y ffmpeg build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["stdbuf", "-oL", "python", "app.py"]