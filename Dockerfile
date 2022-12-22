# Pulls python image from dockerhub
FROM python:3.9

WORKDIR /usr/app

# Adds the relevant files
COPY json /usr/app/json
COPY main_roberta.py /usr/app/
COPY evaluate-v2.0.py /usr/app/

# Installs all the libs
#RUN pip install collections logging json
#RUN pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install transformers==2.5.1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers torch numpy

#RUN pip install --no-cache-dir --upgrade pip && \
#   pip install --no-cache-dir collections logging json torch torchvision transformers


# Runs the code
#CMD ['python new_main.py'; 'python evaluate-v2.0.py json/dev-v2.0.json json/predictions.json' ]
#CMD [ "python", "./evaluate-v2.0.py json/dev-v2.0.json json/preds_roberta.json" ]
CMD ["python", "./main_roberta.py"]