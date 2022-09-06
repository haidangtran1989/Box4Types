import torch

from models.box_model_classifier import BoxModelClassifier

model = BoxModelClassifier()
checkpoint = torch.load("../model/box_ufet.pt")
model.load_state_dict(checkpoint["state_dict"], strict=False)
text = [["Man United", "Man United is a football club in the UK"],
        ["their", "Man United is a football club in the UK.. what are their trophies"],
        ["Antony", "Manchester United has completed the transfer of Antony from Ajax. He has signed a contract until 2027, with the option of an additional year."],
        ["Tyrell Malacia", "Manchester United is pleased to confirm that Tyrell Malacia has joined the club, signing a contract until June 2026, with the option to extend for a further year."],
        ["Dell", "Dell is an American multinational technology company that develops, sells, repairs, and supports computers and related products and services and is owned by its parent company, Dell Technologies."],
        ["Dell", "Michael Dell is an American billionaire businessman and philanthropist. He is the founder, chairman, and CEO of Dell Technologies, one of the world's largest technology infrastructure companies."]]
types = model.classify(text)
for x in types:
    print(x)

import json

class Example:
    def __init__(self, ex_id, right_context_text, left_context_text, word):
        self.ex_id = ex_id
        self.right_context = right_context_text.split(" ")
        self.left_context = left_context_text.split(" ")
        self.right_context_text = right_context_text
        self.left_context_text = left_context_text
        self.y_category = []
        self.word = word
        self.mention_as_list = word.split(" ")

count = 0
for pair in text:
    mention = " " + pair[0] + " "
    context = " " + pair[1] + " "
    idx = context.find(mention)
    prefix = context[0:idx].strip()
    suffix = context[idx + len(mention):].strip()
    count += 1
    example = Example(str(count), suffix, prefix, mention.strip())
    x = json.dumps(example.__dict__)
    print(x)
