import os
import jieba
import torch
import collections
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

from config import Config


class CustomDataset(Dataset):
    def __init__(self, config):
        train_in = CustomDataset.load_from_file(os.path.join(config.train_data_dir, "in.txt"))
        train_out = CustomDataset.load_from_file(os.path.join(config.train_data_dir, "out.txt"))
        valid_in = CustomDataset.load_from_file(os.path.join(config.valid_data_dir, "in.txt"))
        valid_out = CustomDataset.load_from_file(os.path.join(config.valid_data_dir, "out.txt"))
        assert len(train_in) == len(train_out)
        assert len(valid_in) == len(valid_out)
        self.char2id = config.char2id
        self.word2id = config.word2id
        # 分词 + 转化为ID序列
        for datas in [train_in, train_out, valid_in, valid_out]:
            CustomDataset.segment(datas)
            self.seq_to_ids(datas)
        self.train_data = [train_in, train_out]
        self.valid_data = [valid_in, valid_out]

    def __len__(self):
        return len(self.train_data[0])

    def __getitem__(self, idx):
        return [self.train_data[0][idx], self.train_data[1][idx]]

    @staticmethod
    def load_from_file(path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as fr:
            datas = ["".join(line.strip().split(' ')) for line in fr if line.strip()]
        return datas

    @staticmethod
    def jieba_segment(text: str, mode: str) -> List[str]:
        seg = list(jieba.tokenize(text, mode=mode))
        # build DAG
        graph = collections.defaultdict(list)
        for word in seg:
            graph[word[1]].append(word[2])

        def dfs(graph: Dict[int, List[int]], v: int, seen: List[int] = None, path: List[int] = None):
            if seen is None:
                seen = []
            if path is None:
                path = [v]
            seen.append(v)
            paths = []
            for t in graph[v]:
                if t not in seen:
                    t_path = path + [t]
                    paths.append(tuple(t_path))
                    paths.extend(dfs(graph, t, seen, t_path))
            return paths

        longest_path = sorted(dfs(graph, 0), key=lambda x: len(x), reverse=True)[0]
        longest_seg = [text[longest_path[i]: longest_path[i + 1]] for i in range(len(longest_path) - 1)]
        return longest_seg

    @staticmethod
    def segment(datas):
        for i in tqdm(range(len(datas)), desc='预分词'):
            text = datas[i]
            datas[i] = [CustomDataset.jieba_segment(text, mode='default')]

    def seq_to_ids(self, datas):
        for i in range(len(datas)):
            char_ids = []
            word_ids = []
            seg = datas[i][0]
            for t in seg:
                word_ids.extend([self.word2id.get(t, len(word_ids)) * len(t)])  # last index stand for UNK
                for c in t:
                    char_ids.append(self.char2id.get(c, len(char_ids)))
            datas[i].append(char_ids)
            datas[i].append(word_ids)

    def custom_collate_fn(self, batch):
        tensor_in = []
        tensor_out = []
        max_length_in = -1
        max_length_out = -1
        for data in batch:
            data_in, data_out = data
            # char level ids only
            tensor_in.append(data_in[1])
            tensor_out.append(data_out[1])
            max_length_in = max(max_length_in, len(data_in[1]))
            max_length_out = max(max_length_out, len(data_out[1]))
        tensor_in = [arr[:max_length_in] if len(arr) >= max_length_in else arr + [0] * (max_length_in-len(arr)) for arr in tensor_in]
        tensor_mask = [[1 if v != 0 else 0 for v in arr] for arr in tensor_in]
        tensor_out = [arr[:max_length_out] if len(arr) >= max_length_out else arr + [0] * (max_length_out-len(arr)) for arr in tensor_out]
        tensor_in = torch.Tensor(np.array(tensor_in)).long()
        tensor_mask = torch.Tensor(np.array(tensor_mask)).long()
        tensor_out = torch.Tensor(np.array(tensor_out)).long()
        return (tensor_in, tensor_mask), tensor_out


if __name__ == "__main__":
    config = Config("resources/couplet/vocabs")
    dataset = CustomDataset(config)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.custom_collate_fn)
    for data in dataloader:
        print(1)
