class Config:
    def __init__(self, char_table_file):
        self.train_data_dir = "resources/couplet/train_mini"
        self.valid_data_dir = "resources/couplet/test"
        self.learning_rate = 0.01
        self.batch_size = 64
        self.epoch = 50
        self.dropout_level = 0.1
        self.use_mix_embedding = False
        self.char_dim = 64
        self.word_dim = 100
        self.hidden_dim = 128
        self.kernel_size = 3
        self.dilation = 1
        with open(char_table_file, 'r', encoding='utf-8') as fr:
            # 0 for padding, last for UNK
            self.char2id = {char.strip().split('\t')[0]: i+1 for i, char in enumerate(fr) if int(char.strip().split('\t')[1]) >= 10}
            self.id2char = {i: char for char, i in self.char2id.items()}
        self.char_table_size = len(self.char2id)
        self.word2id = {}
        self.word_table_size = 0

        self.output_dir = "result/model"
