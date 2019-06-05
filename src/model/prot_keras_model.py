from keras.models import Model
from keras.layers import Input, Lambda
from keras import optimizers
import keras.backend as K
from src.model.keras_model import STModel
from src.architectures.prot_architectures import prot_encoder
from src.architectures.pred_architectures import FNN_pred
from src.utils.prot_utils import NB_AA_ATTRIBUTES, NB_PROT_FEATURES
from src.utils.DB_utils import LIST_AA_DATASETS
from src.utils.package_utils import get_lr_metric, load_model_weights


def permute_all_axes(x, pattern):
    return K.permute_dimensions(x, pattern)


class ProtModel(STModel):

    def _init_encoder(self, enc_dict_param):
        self.list_type_unfreeze = []
        self.list_nepoch_unfreeze = []
        if enc_dict_param['seq_len'] == 0:
            self.seq_length = None  # None or int = None
        else:
            self.seq_length = enc_dict_param['seq_len']  # None or int
        self.curriculum = enc_dict_param['p_curriculum']  # str ('None' or file path)
        if self.p_curriculum != 'None':
            self.p_epochs_before_trainable = enc_dict_param['p_ne_curriculum']
            self.list_type_unfreeze.append('prot')
            self.list_nepoch_unfreeze.append(enc_dict_param['p_ne_curriculum'])
        self.prot_encoder = enc_dict_param['p_encoder']  # dict with at least 'name' as key
        self.prot_BN = enc_dict_param['BN']  # bool : activate batch normalization or not
        self.prot_dropout = enc_dict_param['dropout']  # float : fraction of the input units droped
        self.prot_reg = enc_dict_param['reg']  # float, weight in l2 reg
        if 'conv_biLSTM' in self.prot_encoder['name']:
            self.prot_encoder['n_steps'] = 1

    def build(self,):
        # input data
        if self.dataset.name in LIST_AA_DATASETS:
            seq = Input(shape=(NB_AA_ATTRIBUTES, 1),
                        dtype='float32', name='in_seq')
            seq_ = Lambda(lambda t: permute_all_axes(t, (2, 1, 0)), name='permute_aa')(seq)
        else:
            seq = Input(shape=(NB_AA_ATTRIBUTES, self.seq_length),
                        dtype='float32', name='in_seq')
            seq_ = Lambda(lambda t: t, name='identity')(seq)
        seq_size = Input(shape=(1,), dtype='float32', name='in_seq_length')
        if self.prot_encoder['hand_crafted_features']:
            self.features = Input(shape=(NB_PROT_FEATURES[self.dataset.name],),
                                  dtype='float32', name='in_hand_crafted_features')
            inputs = [seq, seq_size, self.features]
        else:
            inputs = [seq, seq_size]

        # encoder
        self.embedding, self.emb_size = prot_encoder(self, seq_, seq_size)
        self.embedding = Lambda(lambda t: t, name="embedding")(self.embedding)

        # load encoder weights
        if self.curriculum != 'None':
            load_model_weights(self.model.model, self.curriculum)

        # predictor
        self.output = FNN_pred(self, self.embedding)

        self.optimizer = optimizers.Adam(lr=self.init_lr)
        self.lr_metric = get_lr_metric(self.optimizer)
        self.output_metrics = self.dataset.metrics + [self.lr_metric]
        # compile model
        self.model = Model(inputs=inputs, outputs=[self.output])
        self.model.compile(optimizer=self.optimizer,
                           loss={'output': self.dataset.loss},
                           loss_weights={'output': 1.},
                           metrics={'output': self.output_metrics})
