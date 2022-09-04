from models.transformer_box_model import TransformerBoxModel
from utils.mention_context_similarity import get_mention_context_similarity

model = TransformerBoxModel()
text = [["Trump", "Trump is one of bad presidents of the US"],
        ["Peter Trump", "Peter Trump is a British footballer"]]
rep = model.build_representation_from_texts(text)
sim = get_mention_context_similarity(rep, rep)
print(sim)
