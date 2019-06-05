from keras.layers import Activation, Permute, Dropout, BatchNormalization, Conv2D, Add
from keras.layers import Lambda, Concatenate, InputSpec, Multiply, GaussianNoise
# import keras.regularizers as regularizers
# from keras.layers import dot
import keras.backend as K
from keras.layers import Layer
from src.architectures.GAT_layer import GraphAttention
import tensorflow as tf
# import numpy as np


class RepeatVector4D(Layer):

    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        pattern = K.stack([1, self.n, 1, 1])
        return K.tile(x, pattern)


def dynamic_repeat4D(tensor):
    return RepeatVector4D(K.shape(tensor)[1])(tensor)


def dynamic_repeataxis(tensor, axis, rep=None):
    rep = rep if rep is not None else K.shape(tensor)[axis]
    pattern = [1, 1, 1]
    pattern[axis] = rep
    pattern = K.stack(pattern)
    return K.tile(tensor, pattern)


# def personbond(tensor):
#     import pdb; pdb.Pdb().set_trace()
#     return K.batch_dot(tensor, Aaug, axes=2)


def norm_mat_layer(adj, n_filters):
    norm = K.repeat(K.sum(adj, axis=2, keepdims=False), n_filters)
    norm = K.expand_dims(Permute((2, 1))(norm), axis=-2)
    return norm


def batch_dot_layer(_x1, _x2, axes):
    return K.batch_dot(_x1, _x2, axes)


def expand_last_dim(input_tensor):
    return expand_dim(input_tensor, axis=-1)


def expand_dim(input_tensor, axis):
    return K.expand_dims(input_tensor, axis=axis)


def squeeze(input_tensor, axis):
    return K.squeeze(input_tensor, axis)


def agg_sum(input_tensor):
    return K.sum(input_tensor, axis=1, keepdims=False)


def agg_max(input_tensor):
    return K.max(input_tensor, axis=1, keepdims=False)


def GAT_func(_emb_atoms, n_filters, n_heads, adj, agg_nei, reg, drop):
    # emb_shape = _emb_atoms.get_shape().as_list()
    # kernel_size, strides = (1, emb_shape[2]), (1, emb_shape[2])
    # if reg != 0:
    #     kreg, breg = regularizers.l2(reg), regularizers.l2(reg)
    # else:
    #     kreg, breg = None, None

    _emb_atoms = GraphAttention(n_filters, n_heads, dropout_rate=drop,
                                kernel_regularizer=None, bias_regularizer=None,
                                attn_kernel_regularizer=None)([_emb_atoms, adj])
    # import pdb; pdb.Pdb().set_trace()
    _emb_atoms = Lambda(lambda t: t, output_shape=(None, n_filters * n_heads))(_emb_atoms)
    # emb_atoms = Lambda(lambda t: squeeze(t, -2),
    #                    output_shape=(None, n_filters * n_heads))(emb_atoms)
    #
    return _emb_atoms


def standard_func(model, _emb_atoms, n_filters, adj, reg, drop, step, prefix):
    emb_shape = _emb_atoms.get_shape().as_list()
    # print(emb_shape, adj.shape)

    kernel_size, strides = (1, emb_shape[2]), (1, emb_shape[2])
    kreg, breg = None, None
    # if reg != 0:
    #     kreg, breg = regularizers.l2(reg), regularizers.l2(reg)
    # else:
    #     kreg, breg = None, None

    x1 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
                padding='same', data_format='channels_last',
                activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
                name=str(step) + prefix + '_emb_conv')(Lambda(lambda t: expand_last_dim(t))
                                                       (_emb_atoms))
    if reg != 0:
        x1 = GaussianNoise(reg)(x1)
    x1 = Lambda(lambda t: squeeze(t, -2), name=str(step) + prefix + "_squeeze_x1")(x1)

    ##### new
    # print("1 x2", emb_atoms)
    x2_a = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', data_format='channels_last',
                  activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
                  name=str(step) + prefix + '_emb_nei')(Lambda(lambda t: expand_last_dim(t))
                                                        (_emb_atoms))
    x2_b = Lambda(lambda t: squeeze(t, -2), name=str(step) + prefix + "_squeeze_x2")(x2_a)
    # print('2 x2', x2_b)
    x2_c = Lambda(lambda arg: batch_dot_layer(arg[0], arg[1], axes=(2, 1)),
                  name=str(step) + prefix + "_emb_nei_sum")([adj, x2_b])
    if reg != 0:
        x2_c = GaussianNoise(reg)(x2_c)
    # print('3 x2', x2_c)
    # print("x1", x1)

    ##### old
    # x2 = Lambda(lambda arg: batch_dot_layer(arg[0], arg[1], (2, 1)),
    #             name=str(step) + '_emb_nei_sum')([adj, emb_atoms])
    # print('1 x2', x2)
    # x2 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
    #             padding='same', data_format='channels_last',
    #             activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
    #             name=str(step) + '_nei_conv')(Lambda(lambda t: expand_last_dim(t))(x2))
    # print('2 x2', x2)
    # x2 = Multiply()([norm, x2])  # , name=str(step) + '_emb_nei_sum_norm')
    # print('3 x2', x2)

    # x3 = x1 + x2 ; x3 = K.squeeze(ReLU()(x3), axis=-2) ; print('x3.shape', x3.shape)
    emb_atoms = \
        (Activation('relu', name=str(step) + prefix + '_emb_atoms')(Add(name="")([x1, x2_c])))
    if drop != 0:
        emb_atoms = Dropout(drop, noise_shape=None, seed=None)(emb_atoms)
    else:
        print('mol drop is 0 ###########')
    return emb_atoms


def standard_func_wbond(model, _emb_atoms, _emb_bonds, n_filters, adj, reg, drop, step, prefix):
    emb_shape = _emb_atoms.get_shape().as_list()
    bond_shape = _emb_bonds.get_shape().as_list()
    # print(emb_shape, bond_shape, adj.shape)

    n_filters = n_filters
    kernel_size_1, kernel_size_2 = (1, emb_shape[2]), (1, bond_shape[2] + emb_shape[2])
    strides_1, strides_2 = (1, emb_shape[2]), (1, bond_shape[2] + emb_shape[2])
    kreg, breg = None, None
    # if reg != 0:
    #     kreg, breg = regularizers.l2(reg), regularizers.l2(reg)
    # else:
    #     kreg, breg = None, None

    x1 = Conv2D(filters=n_filters, kernel_size=kernel_size_1, strides=strides_1,
                padding='same', data_format='channels_last',
                activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
                name=str(step) + '_emb_conv')(Lambda(lambda t: expand_last_dim(t))(_emb_atoms))
    if reg != 0:
        x1 = GaussianNoise(reg)(x1)
    x1 = Lambda(lambda t: squeeze(t, -2))(x1)

    # xbis = RepeatVector4D(emb_bonds.shape[1])(emb_atoms)
    # xbis = Lambda(lambda t: dynamic_repeat4D(t),
    #               output_shape=(None, None, emb_shape[2]))(emb_atoms)
    xbis = Lambda(lambda t: dynamic_repeataxis(t, 1, rep=None),
                  name=str(step) + 'x2repeat')(_emb_atoms)
    # print("xbis", xbis)
    x2 = Concatenate(axis=-1,
                     name=str(step) + 'x2concat')([xbis, _emb_bonds])
    # print("1 x2", x2)
    x2bis = Lambda(lambda t: expand_last_dim(t), name=str(step) + 'x2expand')(x2)
    # print("1bis x2", x2bis)
    x2_a = Conv2D(filters=n_filters, kernel_size=kernel_size_2, strides=strides_2,
                  padding='same', data_format='channels_last',
                  activation=None, kernel_regularizer=kreg, bias_regularizer=breg,
                  name=str(step) + 'x2_emb_nei')(x2bis)

    x2_b = Lambda(lambda t: squeeze(t, -2), name=str(step) + 'x2squeeze')(x2_a)
    # print('2 x2', x2_b)
    x2_c = Lambda(lambda args: K.batch_dot(args[0], args[1], axes=(2, 1)))([adj, x2_b])
    # print('3 x2', x2_c)
    # print("x1", x1)
    if reg != 0:
        x2_c = GaussianNoise(reg)(x2_c)

    emb_atoms = Activation('relu')(Add()([x1, x2_c]))
    if drop != 0:
        emb_atoms = Dropout(drop, noise_shape=None, seed=None)(emb_atoms)
    # print('emb_atoms', emb_atoms)
    return emb_atoms


def mol_encoder_agg_nei(model, agg_nei, emb_atoms, emb_bonds, adj, norm, step, bn, drop, reg):
    if agg_nei['name'] == "standard":
        emb_atoms = standard_func(model, emb_atoms, agg_nei['n_filters'], adj, reg, drop, step, '')
        # print('emb_atoms', emb_atoms)

    elif agg_nei['name'] == 'standard_wbond':
        emb_atoms = standard_func_wbond(model, emb_atoms, emb_bonds, agg_nei['n_filters'], adj,
                                        reg, drop, step, '')

    elif agg_nei['name'] == 'GAT':
        emb_atoms = GAT_func(emb_atoms, agg_nei['n_filters'], agg_nei['n_heads'],
                             adj, agg_nei, reg, drop)

    if bn:
        emb_atoms = BatchNormalization(axis=2, name=str(step) + '_aggBN')(emb_atoms)
    else:
        print("no mol BN ########")
    return emb_atoms, emb_bonds, adj


def mol_encoder_agg_all(model, agg_all, new_emb_atoms, emb_atoms, emb_bonds, adj, step, nb_step):
    if agg_all['name'] == "sum":
        temp_emb = Lambda(lambda t: agg_sum(t), name=str(step) + '_emb_mol')(new_emb_atoms)
    elif agg_all['name'] == "max":
        temp_emb = Lambda(lambda t: agg_max(t), name=str(step) + '_emb_mol')(new_emb_atoms)
    elif agg_all['name'] == "pool":
        temp_emb = None
        n_nodes = int(agg_all['n_atoms'] / (2**(step + 1)))
        reg, drop = agg_all['reg'], agg_all['drop']

        # assignment matrix (n_l * n_l+1)
        print(step, nb_step)
        if step + 1 == nb_step:
            temp_emb = Lambda(lambda t: agg_sum(t), name=str(step) + '_emb_mol')(new_emb_atoms)
        else:
            if agg_all['pool_name'] == 'standard':
                S = standard_func(model, emb_atoms, n_nodes, adj, reg, drop, step, 'assign')
            S = Lambda(lambda t: K.softmax(t, axis=2), name=str(step) + "_assign_softmax")(S)
            model.assign_loss += K.mean(tf.norm(adj - K.batch_dot(S, Permute((2, 1))(S)),
                                                ord='fro', axis=[-2, -1]),
                                        axis=0, keepdims=False)
            print('################', model.assign_loss)

            new_emb_atoms = Lambda(lambda arg: K.batch_dot(Permute((2, 1))(arg[0]), arg[1]),
                                   name=str(step) + '_new_emb')([S, new_emb_atoms])
            adj = Lambda(lambda arg: K.batch_dot(K.batch_dot(Permute((2, 1))(arg[0]), arg[1]),
                                                 arg[0]), name=str(step) + '_new_adj')([S, adj])
            # print(adj)
            # print(K.log(adj))
            # print(Multiply()([adj, K.log(adj)]))
            # model.H_loss += K.mean(-1 * K.sum(Multiply()([adj, K.log(adj)]),
            #                                   axis=2, keepdims=False),
            #                        axis=[0, 1], keepdims=False)
            print('################', model.H_loss)

        # import pdb; pdb.Pdb().set_trace()

    return temp_emb, new_emb_atoms, emb_bonds, adj


def mol_encoder_combine(combine, mol_reprentations):
    if combine['name'] == 'sum':
        embedding = Add(name="mol_embedding")(mol_reprentations)
    elif combine['name'] == 'concat':
        embedding = Concatenate(axis=-1, name="mol_embedding")(mol_reprentations)
    elif combine['name'] == 'last':
        embedding = Lambda(lambda t: t, name="mol_embedding")(mol_reprentations[-1])
    return embedding


def mol_encoder(model, emb_atoms, emb_bonds, adj):
    mol_reprentations = []

    if "standard" in model.agg_nei['name']:
        norm = Lambda(lambda t: norm_mat_layer(t, model.agg_nei['n_filters']),
                      name="norm_layer")(adj)
    else:
        norm = None
    # print('norm', norm)

    for n in range(model.n_steps):
        new_emb_atoms, emb_bonds, adj = \
            mol_encoder_agg_nei(model, model.agg_nei, emb_atoms, emb_bonds, adj,
                                norm, n, model.mol_BN, model.mol_dropout, model.mol_reg)
        temp_emb, emb_atoms, emb_bonds, adj = \
            mol_encoder_agg_all(model, model.agg_all, new_emb_atoms, emb_atoms, emb_bonds, adj, n,
                                model.n_steps)
        mol_reprentations.append(temp_emb)

    embedding = mol_encoder_combine(model.combine, mol_reprentations)
    if model.combine['hand_crafted_features']:
        embedding = Concatenate(axis=-1)([embedding, model.features])

    return embedding
