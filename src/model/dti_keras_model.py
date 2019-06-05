from keras.models import Model
from keras.layers import Input, Lambda
from keras import optimizers
from src.model.keras_model import STModel, MTModel
from src.architectures.mol_architectures import mol_encoder
from src.architectures.prot_architectures import prot_encoder
from src.architectures.dti_architectures import dti_jointer
from src.architectures.pred_architectures import FNN_pred
from src.utils.prot_utils import NB_AA_ATTRIBUTES, NB_PROT_FEATURES
from src.utils.mol_utils import NB_ATOM_ATTRIBUTES, NB_BOND_ATTRIBUTES, NB_MOL_FEATURES
from src.utils.package_utils import get_lr_metric, load_model_weights, get_lossw_metric
from src.utils.package_utils import unfreeze_callback, load_model_pred_weights
from src.utils.package_utils import load_FFN_weights
import keras.backend as K


class DTIModel(STModel):

    def _init_encoder(self, enc_dict_param):
        self.list_type_unfreeze = []
        self.list_nepoch_unfreeze = []

        if enc_dict_param['seq_len'] == 0:
            self.seq_length = None  # None or int = None
        else:
            self.seq_length = enc_dict_param['seq_len']  # None or int
        self.p_curriculum = enc_dict_param['p_curriculum']  # str ('None' or file path)
        if self.p_curriculum != 'None':
            self.p_epochs_before_trainable = enc_dict_param['p_ne_curriculum']
            self.list_type_unfreeze.append('prot')
            self.list_nepoch_unfreeze.append(enc_dict_param['p_ne_curriculum'])
        else:
            self.p_epochs_before_trainable = 0
        self.prot_encoder = enc_dict_param['p_encoder']  # dict with at least 'name' as key
        self.prot_encoder['hand_crafted_features'] = 0
        self.prot_BN = enc_dict_param['p_BN']  # bool : activate batch normalization or not
        self.prot_dropout = enc_dict_param['p_drop']  # float : fraction of the input units droped
        self.prot_reg = enc_dict_param['p_reg']  # float, weight in l2 reg

        # MOLECULE ENCODER
        if enc_dict_param['n_att_atom'] == 0:
            self.n_att_atom = None
        else:
            self.n_att_atom = enc_dict_param['n_att_atom']  # None or int
        self.m_curriculum = enc_dict_param['m_curriculum']  # str ('None' or file path)
        if self.m_curriculum != 'None':
            self.m_epochs_before_trainable = enc_dict_param['m_ne_curriculum']
            self.list_type_unfreeze.append('mol')
            self.list_nepoch_unfreeze.append(enc_dict_param['m_ne_curriculum'])
        else:
            self.m_epochs_before_trainable = 0
        self.n_steps = enc_dict_param['m_n_steps']  # int (nb of conv)
        self.agg_nei = enc_dict_param['agg_nei']  # dict : contains at least "name"
        self.agg_all = enc_dict_param['agg_all']  # idem
        self.combine = enc_dict_param['combine']  # idem
        self.combine['hand_crafted_features'] = 0
        self.mol_BN = enc_dict_param['m_BN']  # bool : activate batch normalization or not
        self.mol_dropout = enc_dict_param['m_drop']  # float : fraction of the input units to drop
        self.mol_reg = enc_dict_param['m_reg']  # float, weight in l2 reg
        if self.agg_all['name'] == 'pool':
            self.assign_loss = 0.
            self.H_loss = 0.

        # PREDICTOR
        self.pred_curriculum = enc_dict_param['pred_curriculum']  # str ('None' or file path)
        if self.pred_curriculum != 'None':
            self.pred_epochs_before_trainable = enc_dict_param['pred_ne_curriculum']
            self.list_type_unfreeze.append('pred')
            self.list_nepoch_unfreeze.append(enc_dict_param['pred_ne_curriculum'])
        else:
            self.pred_epochs_before_trainable = 0

        # DTI JOINTER
        self.jointer = enc_dict_param['joint']
        if self.jointer['name'] == 'combine' and self.prot_encoder['name'] == "conv_biLSTM":
            print('WARNING: prot encoder is changed to "conv_biLSTM_aa"')
            self.prot_encoder['name'] = "conv_biLSTM_aa"
        elif self.jointer['name'] == 'combine' and self.prot_encoder['name'] == "conv":
            print('WARNING: prot encoder is changed to "conv_aa"')
            self.prot_encoder['name'] = "conv_aa"
        if self.jointer['name'] == 'combine' and self.prot_encoder['name'] == "conv_biLSTM_aa" and\
                self.prot_encoder['nb_conv_filters'] * 2 != self.agg_nei['n_filters']:
            print('WARNING: nb_filters do not fit')
            self.agg_nei['n_filters'] = 2 * self.prot_encoder['nb_conv_filters']

    def build(self,):
        # PROTEIN input data
        seq = Input(shape=(NB_AA_ATTRIBUTES, self.seq_length),
                    dtype='float32', name='in_seq')
        seq_size = Input(shape=(1,), dtype='float32', name='in_seq_length')
        bond_size = self.n_att_atom * self.n_att_atom if self.n_att_atom is not None else None
        prot_inputs = [seq, seq_size]

        # MOLECULE input data
        atom_att = Input(shape=(self.n_att_atom, NB_ATOM_ATTRIBUTES),
                         dtype='float32', name='in_atom_att')
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

        if self.jointer['hand_crafted_features'] > 0:
            # self.prot_features = Input(shape=(NB_PROT_FEATURES[self.dataset.name],),
            #                            dtype='float32', name='in_prot_hand_crafted_features')
            self.prot_features = Input(shape=(NB_PROT_FEATURES,),
                                       dtype='float32', name='in_prot_hand_crafted_features')
            self.mol_features = Input(shape=(NB_MOL_FEATURES,), dtype='float32',
                                      name='in_mol_hand_crafted_features')
            inputs = mol_inputs + [self.mol_features] + prot_inputs + [self.prot_features]
        else:
            inputs = mol_inputs + prot_inputs

        # PROTEIN encoder
        self.prot_embedding, self.emb_size = prot_encoder(self, seq, seq_size)
        # MOLECULE encoder
        self.mol_embedding = mol_encoder(self, atom_att, bond_att, adj)
        # JOINT encoder
        self.embedding = dti_jointer(self, self.prot_embedding, self.mol_embedding)
        self.embedding = Lambda(lambda t: t, name="embedding")(self.embedding)

        # predictor
        if self.jointer['hand_crafted_features'] not in [200, 201, 202, 203, 204, 205]:
            self.output = FNN_pred(self, self.embedding)
        else:
            self.output = Lambda(lambda t: t, name="output")(self.embedding)

        self.optimizer = optimizers.Adam(lr=self.init_lr)
        self.lr_metric = get_lr_metric(self.optimizer)
        self.output_metrics = self.dataset.metrics + [self.lr_metric]
        # if self.agg_all['name'] == 'pool':
        #     print('##### add losses')
        #     self.model.add_loss(self.assign_loss)
        #     self.model.add_loss(self.H_loss)

        # load encoder weights
        self.model = Model(inputs=inputs, outputs=[self.output])

        # else:
        #     self.model = Model(inputs=inputs, outputs=[self.output])
        if self.m_curriculum != 'None':
            load_model_weights(self.model, self.m_curriculum)
            # for layer in self.m_curriculum_trainable_layers:
            #     layer.trainable = False
            #     print(layer.name, layer.trainable)
            if self.m_epochs_before_trainable != 0:
                for layer in self.model.layers:
                    name = layer.name
                    if ('emb_conv' in name or 'emb_nei' in name or 'agg' in name or \
                            'emb_conv' in name) and 'BN' not in name:
                        layer.trainable = False
        if self.p_curriculum != 'None':
            load_model_weights(self.model, self.p_curriculum)
            # for layer in self.p_curriculum_trainable_layers:
            #     layer.trainable = False
            #     print(layer.name, layer.trainable)
            if self.p_epochs_before_trainable != 0:
                for layer in self.model.layers:
                    name = layer.name
                    if ('pconv' in name or 'biLSTM' in name) and 'BN' not in name:
                        layer.trainable = False
        if self.pred_curriculum != 'None':
            load_model_pred_weights(self.model, self.pred_curriculum)
            if self.pred_epochs_before_trainable != 0:
                for layer in self.model.layers:
                    name = layer.name
                    if ('pred_layer' in name or 'BNpred' in name or 'output' in name) and \
                            'BN' not in name:
                        layer.trainable = False
            # for layer in self.pred_curriculum_trainable_layers:
            #     layer.trainable = False
            #     print(layer.name, layer.trainable)

        if self.jointer['hand_crafted_features'] > 1:
            if self.jointer['hand_crafted_features'] in [11, 12, 13]:
                # self.model = Model(inputs=[self.mol_features, self.prot_features],
                #                    outputs=[self.output])
                for layer in self.model.layers:
                    if 'fea_layer' not in layer.name and 'pred' not in layer.name and \
                            'output' not in layer.name:
                        layer.trainable = False
                    print(layer.name, layer.trainable)

            if self.jointer['hand_crafted_features'] == 100:
                self.list_nepoch_unfreeze = []
                load_FFN_weights(self.model, self.jointer['filename'])
                for layer in self.model.layers:
                    if 'pred' not in layer.name and 'output' not in layer.name:
                        layer.trainable = False
                    if 'BN' in layer.name:
                        layer.trainable = True
                    print(layer.name, layer.trainable)

        # compile model
        self.model.compile(optimizer=self.optimizer,
                           loss={'output': self.dataset.loss},
                           loss_weights={'output': 1.},
                           metrics={'output': self.output_metrics})

        # import pdb; pdb.Pdb().set_trace()

        # for layer in self.model.layers:
        #     layer.trainable = False

        print('\nTRAINABLE LAYERS')
        for layer in self.model.layers:
            if layer.trainable:
                print(layer.name, layer.trainable)
        print('\nUN-TRAINABLE LAYERS')
        for layer in self.model.layers:
            if not layer.trainable:
                print(layer.name, layer.trainable)
        # from keras.utils import plot_model
        # plot_model(self.model, to_file='model_aa.png')


class MTDTIModel(MTModel, DTIModel):

    def get_callbacks(self, val_gen, val_y, train_gen, train_val):
        return MTModel.get_callbacks(self, val_gen, val_y, train_gen, train_val)

    def fit(self, train_gen, val_gen, train_val, val_y, ratio_tr, debug=False):
        MTModel.fit(self, train_gen, val_gen, train_val, val_y, ratio_tr, debug)

    def predict(self, test_gen):
        return MTModel.predict(self, test_gen)

    def _init_encoder(self, enc_dict_param):
        DTIModel._init_encoder(self, enc_dict_param)

        # MT PARAMETER
        self.MT_type = enc_dict_param['MT_type']
        self.MT_lossw = enc_dict_param['MT_lossw']

    def build(self,):
        # DTI PROTEIN input data
        seq = Input(shape=(NB_AA_ATTRIBUTES, self.seq_length),
                    dtype='float32', name='in_seq')
        seq_size = Input(shape=(1,), dtype='float32', name='in_seq_length')
        bond_size = self.n_att_atom * self.n_att_atom if self.n_att_atom is not None else None

        # DTI MOLECULE input data
        dti_atom_att = Input(shape=(self.n_att_atom, NB_ATOM_ATTRIBUTES),
                             dtype='float32', name='in_atom_att')
        dti_bond_att = Input(shape=(bond_size, NB_BOND_ATTRIBUTES),
                             dtype='float32', name='in_bond_att')
        if 'bond' in self.agg_nei['name']:
            dti_adj = Input(shape=(self.n_att_atom, bond_size),
                            dtype='float32', name='in_adj')
        else:
            dti_adj = Input(shape=(self.n_att_atom, self.n_att_atom),
                            dtype='float32', name='in_adj')

        # PCBA MOLECULE input data
        pcba_atom_att = Input(shape=(self.n_att_atom, NB_ATOM_ATTRIBUTES),
                              dtype='float32', name='in_atom_att')
        pcba_bond_att = Input(shape=(bond_size, NB_BOND_ATTRIBUTES),
                              dtype='float32', name='in_bond_att')
        if 'bond' in self.agg_nei['name']:
            pcba_adj = Input(shape=(self.n_att_atom, bond_size),
                             dtype='float32', name='in_adj')
        else:
            pcba_adj = Input(shape=(self.n_att_atom, self.n_att_atom),
                             dtype='float32', name='in_adj')
        inputs = [dti_atom_att, dti_bond_att, dti_adj, pcba_atom_att, pcba_bond_att, pcba_adj,
                  seq, seq_size]

        # PROTEIN encoder
        self.prot_embedding, self.emb_size = prot_encoder(self, seq, seq_size)
        # MOLECULE encoder
        atom_att, bond_att, adj = \
            [dti_atom_att, pcba_atom_att], [dti_bond_att, pcba_bond_att], [dti_adj, pcba_adj]
        if self.MT_type == 'hard':
            self.dti_mol_embedding, self.pcba_embedding = \
                hard_mol_encoder(self, atom_att, bond_att, adj)
        elif self.MT_type == 'partial':
            self.dti_mol_embedding, self.pcba_embedding = \
                partial_mol_encoder(self, atom_att, bond_att, adj)
        self.pcba_embedding = Lambda(lambda t: t, name="pcba_embedding")(self.pcba_embedding)
        # JOINT encoder
        self.dti_embedding = dti_jointer(self, self.prot_embedding, self.dti_mol_embedding)
        self.dti_embedding = Lambda(lambda t: t, name="dti_embedding")(self.dti_embedding)

        # predictor
        dti_output = FNN_pred(self, self.dti_embedding, 'dti')
        pcba_output = FNN_pred(self, self.pcba_embedding, 'pcba')
        self.output = [dti_output, pcba_output]

        optimizer = optimizers.Adam(lr=self.init_lr)
        lr_metric = get_lr_metric(optimizer)
        lossw_metric = get_lossw_metric(self.dti_lossw, self.pcba_lossw)
        # compile model
        self.model = Model(inputs=inputs, outputs=self.output)

        self.dti_lossw = K.variable(0.5)
        self.pcba_lossw = K.variable(0.5)
        self.model.compile(optimizer=optimizer,
                           loss={'dti_output': self.dataset.loss[0],
                                 'pcba_output': self.dataset.loss[1]},
                           loss_weights={'dti_output': self.dti_lossw,
                                         'pcba_output': self.pcba_lossw},
                           metrics={'dti_output': self.dataset.metrics + [lr_metric],
                                    'pcba_output': self.dataset.metrics + [get_lossw_metric]})
