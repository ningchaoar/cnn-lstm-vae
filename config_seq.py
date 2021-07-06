import torch


class Config:
    def __init__(self, char_table_file):
        self.train_data_dir = "resources/couplet/train_mini"
        self.valid_data_dir = "resources/couplet/test"
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epoch = 50
        self.dropout_level = 0.1
        self.use_mix_embedding = False
        self.char_dim = 128
        self.word_dim = 100
        self.hidden_dim = 256
        self.max_length = None  # if set None, max length will depend on the longest input in a batch.
        self.model_type = "LSTM"  # "GCNN" -> gated cnn, "LSTM" -> bi-LSTM
        # GCNN settings
        self.kernel_size = 3
        self.dilation = 1
        # LSTM settings
        self.layers = 1

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(char_table_file, 'r', encoding='utf-8') as fr:
            # 0 for padding, last for UNK
            self.char2id = {char.strip().split('\t')[0]: i+1 for i, char in enumerate(fr) if int(char.strip().split('\t')[1]) >= 1}
            self.id2char = {i: char for char, i in self.char2id.items()}
        self.char_table_size = len(self.char2id)  # last index for 'UNK'
        self.word2id = {}
        self.word_table_size = len(self.word2id)

        self.output_dir = "result/model_lstm_b64_c128_h256"
