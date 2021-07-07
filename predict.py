import os
import torch
import numpy as np

from config_seq import Config
from model import CoupletSeqLabeling


class ModelTest:
    def __init__(self):
        self.config = Config("resources/couplet/chars_sort.txt")
        self.model = CoupletSeqLabeling(self.config)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, input):
        ids = torch.Tensor(np.array([self.config.char2id.get(c, self.config.char_table_size + 1) for c in input])).long().unsqueeze(0)
        masks = torch.Tensor(np.array([1] * len(input))).unsqueeze(0)
        lengths = [len(input)]
        logits = self.model((ids, masks, lengths))
        outputs = torch.argmax(logits, dim=2).long()
        predict = "".join([self.config.id2char.get(i, "<UNK>") if i != 0 else " " for i in outputs[0].tolist()])
        print("predict: {}\n".format(predict))


if __name__ == "__main__":
    m = ModelTest()
    model_dir = "result/model_b64_c128_h256_ffw"
    model_files = os.listdir(model_dir)
    while True:
        text = input("type an input string: ")
        for mf in model_files:
            m.load_model(os.path.join(model_dir, mf))
            print(mf, end=" ")
            m.predict(text)
