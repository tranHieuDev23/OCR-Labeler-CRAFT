FROM nvidia/cuda:10.1-base-ubuntu18.04
WORKDIR /app
COPY requirements.txt .
RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "gunicorn craft_api:app -b 0.0.0.0:8000 --workers 4"]