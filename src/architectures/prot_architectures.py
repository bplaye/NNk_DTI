import keras.backend as K
from keras.layers import Bidirectional, Masking, LSTM, Conv2D, Permute, Lambda, GRU
from keras.layers import Dropout, BatchNormalization, Dense, Multiply, Concatenate
from keras.layers import GaussianNoise, Activation
# import keras.regularizers as regularizers
# from src.utils.prot_utils import NB_AA_ATTRIBUTES
from src.utils.DB_utils import LIST_AA_DATASETS


def update_seq_size(seq_size, denominator):
    return K.round(seq_size / denominator)


def slice_prot(x, size_per_sample):
    return x[:, :, :K.cast(size_per_sample, dtype='int32')[0, 0], :]


def expand_last_dim(input_tensor):
    return K.expand_dims(input_tensor, axis=-1)


def agg_sum(input_tensor, axis):
    return K.sum(input_tensor, axis=axis, keepdims=False)


def squeeze(input_tensor, axis):
    return K.squeeze(input_tensor, axis=axis)


def seq_conv(model, seq, n_filters, kernel_size, strides, kreg, breg, noise, drop, bn, step):
    # import pdb; pdb.Pdb().set_trace()
    print(seq)
    seq = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
                 padding='same', data_format='channels_last',
                 activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
                 name=str(step) + '_pconv')(seq)
    if noise != 0:
        seq = GaussianNoise(noise)(seq)
    seq = Activation('relu')(seq)
    seq = Permute((3, 2, 1))(seq)
    print(seq, '########')
    if drop != 0:
        seq = Dropout(drop, noise_shape=None, seed=None)(seq)
    else:
        print('prot drop is 0 ###########')
    if bn:
        seq = BatchNormalization(axis=1, name=str(step) + '_pconvBN')(seq)
    return seq


def prot_encoder(model, seq, seq_size):
    encoder = model.prot_encoder

    if model.dataset.name in LIST_AA_DATASETS:
        if encoder['conv_strides'] != 1 or encoder['name'] == 'conv_biLSTM_att'\
                or model.batch_size != 1:
            print('!! ERROR !!: AA prediction but conv_strides > 1 ' +
                  'OR tt_mech at the end of encoder OR batch_size > 1 !!')
            exit(1)

    kreg, breg = None, None
    # if model.prot_reg != 0:
    #     kreg, breg = regularizers.l2(model.prot_reg), regularizers.l2(model.prot_reg)
    # else:
    #     kreg, breg = None, None

    seq = Lambda(lambda t: expand_last_dim(t), name="expand_last_dim")(seq)

    if encoder['name'] == 'conv' or encoder['name'] == 'conv_aa':
        n_steps = encoder['n_steps']
        n_filters = encoder['nb_conv_filters']
        # seq = Lambda(lambda t: t, name='temp1')(seq)
        for n in range(n_steps):
            seq_shape = seq.get_shape().as_list()
            kernel_size = (seq_shape[1], encoder['filter_size'])
            strides = (seq_shape[1], encoder['conv_strides'])
            seq = seq_conv(model, seq, n_filters, kernel_size, strides, kreg, breg, model.prot_reg,
                           model.prot_dropout, model.prot_BN, n)
            print('seq', seq)
            seq_size = Lambda(lambda t: update_seq_size(t, float(strides[1])))(seq_size)
        # seq = Lambda(lambda t: t, name='temp2')(seq)
        if model.dataset.name not in LIST_AA_DATASETS and 'aa' not in encoder['name']:
            embedding = Lambda(lambda t: agg_sum(t, 2), name='agg_sum')(seq)
            print('###', embedding)
            embedding = Lambda(lambda t: squeeze(t, -1), name='prot_embedding')(embedding)
        elif 'aa' in encoder['name']:
            seq = Lambda(lambda t: squeeze(t, -1), name='conv_emb')(seq)
            seq = Permute((2, 1))(seq)
            embedding = Lambda(lambda t: t, name='prot_embedding')(seq)
        else:
            seq = Permute((2, 1, 3))(seq)
            embedding = Lambda(lambda t: squeeze(squeeze(t, -1), 0), name='prot_embedding')(seq)

    elif 'conv_biLSTM' in encoder['name'] or 'conv_biGRU' in encoder['name']:
        seq_shape = seq.get_shape().as_list()
        n_filters = encoder['nb_conv_filters']
        kernel_size = (seq_shape[1], encoder['filter_size'])
        strides = (seq_shape[1], encoder['conv_strides'])

        seq_size = Lambda(lambda t: update_seq_size(t, float(strides[1])))(seq_size)
        seq = seq_conv(model, seq, n_filters, kernel_size, strides, kreg, breg, model.prot_reg,
                       model.prot_dropout, model.prot_BN, 0)
        seq = Lambda(lambda t: squeeze(t, -1), name='conv_emb')(seq)
        seq = Permute((2, 1))(seq)

        if model.dataset.name not in LIST_AA_DATASETS:
            seq = Masking(mask_value=0.0)(seq)
        rseq = True if 'att' not in encoder['name'] and 'aa' not in encoder['name'] else True

        if 'biLSTM' in encoder['name']:
            lstm_layer = LSTM(n_filters, return_sequences=rseq,
                              activation='tanh', kernel_regularizer=kreg,
                              recurrent_regularizer=kreg, bias_regularizer=breg,
                              dropout=model.prot_dropout)
        elif 'biGRU' in encoder['name']:
            lstm_layer = GRU(n_filters, return_sequences=rseq,
                             activation='tanh', kernel_regularizer=kreg,
                             recurrent_regularizer=kreg, bias_regularizer=breg,
                             dropout=model.prot_dropout)
        # import pdb; pdb.Pdb().set_trace()
        seq = Bidirectional(lstm_layer, merge_mode='concat', weights=None,
                            name='biLSTM')(seq)
        if model.prot_reg != 0:
            seq = GaussianNoise(model.prot_reg)(seq)

        if 'att' in encoder['name']:
            attention_probs = Dense(n_filters * 2, activation='softmax',
                                    name='attention_probs')(seq)
            embedding = Multiply(name='prot_embedding')([seq, attention_probs])

        elif model.dataset.name in LIST_AA_DATASETS or 'aa' in encoder['name']:
            embedding = Lambda(lambda t: t, name='prot_embedding')(seq)
        else:
            embedding = Lambda(lambda t: agg_sum(t, 1), name='prot_embedding')(seq)
            # embedding = Lambda(lambda t: t, name='prot_embedding')(seq)
            # seq = Permute((2, 1))(seq)
            # import pdb; pdb.Pdb().set_trace()
            # embedding = Lambda(lambda t: squeeze(t, 0), name='prot_embedding')(seq)

    if encoder['hand_crafted_features']:
        embedding = Concatenate(axis=-1)([embedding, model.features])
    # import pdb; pdb.Pdb().set_trace()
    return embedding, seq_size
