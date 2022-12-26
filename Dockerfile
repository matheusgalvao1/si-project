# Pulls python image from dockerhub
FROM python:3.9

WORKDIR /usr/app

# Adds the relevant files
COPY data /usr/app/data
COPY predictions /usr/app/predictions
COPY utils /usr/app/utils
COPY requirements.txt /usr/app/requirements.txt
COPY main.py /usr/app/main.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

#RUN pip install --no-cache-dir --upgrade pip && \
#   pip install --no-cache-dir collections logging json torch torchvision transformers

CMD ["python", "./main.py"]