from keras.models import Model
from keras.layers import Input, Lambda
from keras import optimizers
from src.model.keras_model import STModel
from src.architectures.mol_architectures import mol_encoder
from src.architectures.pred_architectures import FNN_pred
from src.utils.mol_utils import NB_ATOM_ATTRIBUTES, NB_BOND_ATTRIBUTES, NB_MOL_FEATURES
from src.utils.package_utils import get_lr_metric, load_model_weights


class MolModel(STModel):

    def _init_encoder(self, enc_dict_param):
        self.curr_callbacks = []
        if enc_dict_param['n_att_atom'] == 0:
            self.n_att_atom = None
        else:
            self.n_att_atom = enc_dict_param['n_att_atom']  # None or int
        self.curriculum = enc_dict_param['m_curriculum']  # none or str (file path)
        if self.m_curriculum != 'None':
            self.p_epochs_before_trainable = enc_dict_param['m_ne_curriculum']
            self.list_type_unfreeze.append('mol')
            self.list_nepoch_unfreeze.append(enc_dict_param['m_ne_curriculum'])
        self.n_steps = enc_dict_param['n_steps']  # int (nb of conv)
        self.agg_nei = enc_dict_param['agg_nei']  # dict : contains at least "name"
        self.agg_all = enc_dict_param['agg_all']  # idem
        self.combine = enc_dict_param['combine']  # idem
        self.mol_BN = enc_dict_param['BN']  # bool : activate batch normalization or not
        self.mol_dropout = enc_dict_param['dropout']  # float : fraction of the input units to drop
        self.mol_reg = enc_dict_param['reg']  # float, weight in l2 reg
        if self.agg_all['name'] == 'pool':
            self.assign_loss = 0.
            self.H_loss = 0.

    def build(self,):
        # input data
        atom_att = Input(shape=(self.n_att_atom, NB_ATOM_ATTRIBUTES),
                         dtype='float32', name='in_atom_att')
        bond_size = self.n_att_atom * self.n_att_atom if self.n_att_atom is not None else None
        if 'bond' in self.agg_nei['name']:
            adj = Input(shape=(self.n_att_atom, bond_size),
                        dtype='float32', name='in_adj')
            bond_att = Input(shape=(bond_size, NB_BOND_ATTRIBUTES),
                             dtype='float32', name='in_bond_att')
            mol_inputs = [atom_att, bond_att, adj]
        else:
            adj = Input(shape=(self.n_att_atom, self.n_att_atom),
                        dtype='float32', name='in_adj')
            bond_att = None
            mol_inputs = [atom_att, adj]
        if self.combine['hand_crafted_features']:
            self.features = Input(shape=(NB_MOL_FEATURES,), dtype='float32',
                                  name='in_hand_crafted_features')
            inputs = mol_inputs + [self.features]
        else:
            inputs = mol_inputs
        print(inputs)

        # encoder
        self.embedding = mol_encoder(self, atom_att, bond_att, adj)
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
        if self.agg_all['name'] == 'pool':
            print('##### add losses')
            self.model.add_loss(self.assign_loss)
            self.model.add_loss(self.H_loss)
        self.model.compile(optimizer=self.optimizer,
                           loss={'output': self.dataset.loss},
                           loss_weights={'output': 1.},
                           metrics={'output': self.output_metrics})
