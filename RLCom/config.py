class Config:
    def __init__(self):
        self.EMB_SIZE = 512
        self.ENC_SIZE = 512
        self.DEC_SIZE = 512
        self.ATTN_SIZE = 512
        self.NUM_LAYER = 2
        self.SEED = 1234
        self.DICT_CODE = 50010
        self.DICT_WORD = 30010
        self.TRAIN_BATCH_SIZE = 32
        self.EVAL_BATCH_SIZE = 16
        self.TEST_BATCH_SIZE = 16
        self.MAX_COMMENT_LEN = 50
        self.MAX_SEQ_LEN = 400
        self.EPOCH = 200
        self.DROPOUT = 0.5
        self.LR = 3e-4
        self.START_TOKEN = 0
        self.END_TOKEN = 1