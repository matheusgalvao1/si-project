# Pulls python image from dockerhub
FROM python:3.9 

# Adds the relevant files
ADD work_SquaD/main.py .

# Installs all the libs
RUN pip install urllib torch transformers wikipedia pprint collections logging json

# Runs the code
CMD [ "python", "./work_SquaD/main.py" ]

