import urllib.request
import torch
import wikipedia as wiki
import pprint as pp
from collections import OrderedDict
import logging
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# executing these commands for the first time initiates a download of the 
# model weights to ~/.cache/torch/transformers/
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2") 
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")




class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.chunked = False
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """ 
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model. 

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1 # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k,v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)
            
            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                
                try:
                    r = self.model(chunk)
                except:
                    r = self.model(**chunk)

                answer_start_scores = r['start_logits']
                answer_end_scores = r['end_logits']
                #answer_start_scores, answer_end_scores = self.model(**chunk)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            r = self.model(**self.inputs)

            answer_start_scores = r['start_logits']
            answer_end_scores = r['end_logits']
            
            answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score
            final_answer = self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])
            if final_answer == '[CLS]':
                final_answer = ''
                
            return final_answer

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))




# to make the following output more readable I'll turn off the token sequence length warning
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


f = open("json/dev-v2.0.json")
data = json.load(f)

answers = {}
reader = DocumentReader("deepset/bert-base-cased-squad2")
cont = 0
for i in data['data']:
    for j in i['paragraphs']:
        for k in j['qas']:
            question = k['question']
            id_q = k['id']
            reader.tokenize(question, j['context'])
            answers[id_q] = reader.get_answer()
            cont += 1
            if(cont%100==0):
                print(f"{cont}, ID: {id_q}")
            
with open("json/predictions.json", "w") as outfile: 
    json.dump(answers, outfile) 