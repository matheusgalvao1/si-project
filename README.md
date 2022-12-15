# si-project
Files:
- evaluate-v2.0.py: Function evaluation already done. It is the official evaluation.
- run_squad.py: File used to train the models. This code already have been developed beacuse it is so complicated to train by ourselves.
- main.ipynb: The main code, that executes all

- dev-v2.0.json: Our test file. Answering these questions, we obtain the evaluation
- example_pred.json: Just a example the file with answers.
- predictions.json: File with out predictions (for the questions in dev-v2.0.json)
- train-v2.0.json: File that we will use to train our models

===============================
To make the predictions, use the main.ipynb. The output of this must be a file like the example_pred.json
To evaluate (calculate F1-score), run this command (in the terminal, inside the project folder):
- python evaluate-v2.0.py dev-v2.0.json predictions.json

