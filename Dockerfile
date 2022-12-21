# Pulls python image from dockerhub
FROM python:3.9 

# Adds the relevant files
ADD json .
ADD new_main.py .
ADD evaluate-v2.0.py .
ADD run_squad.py .

# Installs all the libs
RUN pip install urllib pprint collections logging json
RUN pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==2.5.1
RUN pip install wikipedia==1.4.0

# Runs the code
CMD [ "python", "./new_main.py" ]
CMD [ "python", "./evaluate-v2.0.py json/dev-v2.0.json json/predictions.json" ]
