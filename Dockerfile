FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . .
CMD ["gunicorn", "craft_api:app"]