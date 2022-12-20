# Pulls python image from dockerhub
FROM python:3.9 

# Adds the relevant files
ADD json .
ADD new_main.py .
ADD evaluate-v2.0.py .
ADD run_squad.py .

# Installs all the libs
RUN pip install urllib torch transformers wikipedia pprint collections logging json

# Runs the code
CMD [ "python", "./new_main.py" ]
CMD [ "python", "evaluate-v2.0.py json/dev-v2.0.json json/predictions.json" ]
