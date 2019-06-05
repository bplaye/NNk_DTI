from keras.layers import Concatenate, Dense, Lambda, Activation, Add, Dropout, BatchNormalization
import keras.regularizers as regularizers
import keras.backend as K


def batch_dot_layer(adj, emb_atoms, axes):
    return K.batch_dot(adj, emb_atoms, axes)


def dti_jointer(model, prot_embedding, mol_embedding):
    if model.jointer['name'] == 'concat':
        if model.jointer['hand_crafted_features'] == 1:
            return Concatenate(axis=-1)([prot_embedding, model.prot_features,
                                         mol_embedding, model.mol_features])
        elif model.jointer['hand_crafted_features'] > 1:
            prot_features_emb, mol_features_emb = model.prot_features, model.mol_features
            list_fea = [prot_features_emb, mol_features_emb]
            if model.jointer['hand_crafted_features'] == 50:
                MLP_layers, MLP_drop, MLP_BN = [1000, 500], 0.0, True
            if model.jointer['hand_crafted_features'] == 2:
                MLP_layers, MLP_drop, MLP_BN = [2000, 1000], 0.0, True
            if model.jointer['hand_crafted_features'] == 3:
                MLP_layers, MLP_drop, MLP_BN = [50, 10], 0.0, True
            if model.jointer['hand_crafted_features'] == 4:
                MLP_layers, MLP_drop, MLP_BN = [1000, 500], 0.3, True
            if model.jointer['hand_crafted_features'] == 5:
                MLP_layers, MLP_drop, MLP_BN = [1000, 500], 0.0, True
            if model.jointer['hand_crafted_features'] == 6:
                MLP_layers, MLP_drop, MLP_BN = [1000, 500], 0.6, True
            if model.jointer['hand_crafted_features'] == 7:
                MLP_layers, MLP_drop, MLP_BN = [], 0.0, False
            if model.jointer['hand_crafted_features'] == 8:
                MLP_layers, MLP_drop, MLP_BN = [100], 0.0, True
            if model.jointer['hand_crafted_features'] == 9:
                MLP_layers, MLP_drop, MLP_BN = [500], 0.0, True
            if model.jointer['hand_crafted_features'] == 10:
                MLP_layers, MLP_drop, MLP_BN = [1000], 0.0, True
            if model.jointer['hand_crafted_features'] == 11:
                MLP_layers, MLP_drop, MLP_BN = [2000, 1000], 0.0, True
            if model.jointer['hand_crafted_features'] == 12:
                MLP_layers, MLP_drop, MLP_BN = [100], 0., True
            if model.jointer['hand_crafted_features'] == 100:
                MLP_layers, MLP_drop, MLP_BN = [1000, 500], 0.0, True

            if model.jointer['hand_crafted_features'] in [200, 201, 202, 203, 204, 205]:
                y = Concatenate(axis=-1)([prot_embedding, mol_embedding])
                y = Dense(100, activation='relu',
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          name='nn_layer')(y)
                y = BatchNormalization(axis=-1, name='BNpred_layer')(y)

                MLP_layers, MLP_drop, MLP_BN = [2000, 1000, 100], 0.0, True
                x = Concatenate(axis=-1)([prot_features_emb, mol_features_emb])
                for ilay, lay in enumerate(MLP_layers):
                    x = Dense(lay, activation='relu',
                              kernel_initializer='glorot_uniform', bias_initializer='zeros',
                              name='fea_layer_' + str(ilay))(x)
                    if MLP_drop > 0.0:
                        x = Dropout(MLP_drop, noise_shape=None, seed=None,
                                    name='d_fea_layer_' + str(ilay))(x)
                    if MLP_BN:
                        x = BatchNormalization(axis=-1,
                                               name='bn_fea_layer_' + str(ilay))(x)
                if model.jointer['hand_crafted_features'] == 200:
                    xy = Dense(50, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                elif model.jointer['hand_crafted_features'] == 202:
                    xy = Dense(20, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                    xy = BatchNormalization(axis=-1, name='bn_concat200')(xy)
                elif model.jointer['hand_crafted_features'] == 203:
                    xy = Dense(100, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                    xy = BatchNormalization(axis=-1, name='bn_concat200')(xy)
                elif model.jointer['hand_crafted_features'] == 204:
                    xy = Dense(100, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                elif model.jointer['hand_crafted_features'] == 205:
                    xy = Dense(100, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                    xy = Dense(10, activation='relu', name='concat200_bis')(xy)
                elif model.jointer['hand_crafted_features'] == 201:
                    xy = Dense(100, activation='relu',
                               name='concat200')(Concatenate(axis=-1)([x, y]))
                    xy = BatchNormalization(axis=-1, name='bn_concat200')(xy)
                    xy = Dense(10, activation='relu', name='concat200_bis')(xy)
                    xy = BatchNormalization(axis=-1, name='bn_concat200_bis')(xy)

                return Dense(model.dataset.n_outputs, activation=model.dataset.final_activation,
                             name='200')(xy)

            elif model.jointer['hand_crafted_features'] not in [11, 12]:
                for ilay, lay in enumerate(MLP_layers):
                    for ix, x in enumerate(list_fea):
                        x = Dense(lay, activation='relu',
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  name='fea_layer_' + str(ix) + '_' + str(ilay))(x)
                        if MLP_drop > 0.0:
                            x = Dropout(MLP_drop, noise_shape=None, seed=None)(x)
                        if MLP_BN:
                            x = BatchNormalization(axis=-1)(x)
                        if ix == 0:
                            prot_features_emb = x
                        elif ix == 1:
                            mol_features_emb = x
                    list_fea = [prot_features_emb, mol_features_emb]
                return Concatenate(axis=-1)([prot_embedding, prot_features_emb,
                                             mol_embedding, mol_features_emb])
            elif model.jointer['hand_crafted_features'] == 11:
                x = Concatenate(axis=-1)([prot_features_emb, mol_features_emb])
                for ilay, lay in enumerate(MLP_layers):
                    x = Dense(lay, activation='relu',
                              kernel_initializer='glorot_uniform', bias_initializer='zeros',
                              name='fea_layer_' + str(ilay))(x)
                    if MLP_drop > 0.0:
                        x = Dropout(MLP_drop, noise_shape=None, seed=None,
                                    name='d_fea_layer_' + str(ilay))(x)
                    if MLP_BN:
                        x = BatchNormalization(axis=-1,
                                               name='bn_fea_layer_' + str(ilay))(x)
                return x
            elif model.jointer['hand_crafted_features'] == 12:
                for ilay, lay in enumerate(MLP_layers):
                    for ix, x in enumerate(list_fea):
                        x = Dense(lay, activation='relu',
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  name='fea_layer_' + str(ix) + '_' + str(ilay))(x)
                        if MLP_drop > 0.0:
                            x = Dropout(MLP_drop, noise_shape=None, seed=None)(x)
                        if MLP_BN:
                            x = BatchNormalization(axis=-1)(x)
                        if ix == 0:
                            prot_features_emb = x
                        elif ix == 1:
                            mol_features_emb = x
                    list_fea = [prot_features_emb, mol_features_emb]
                return Concatenate(axis=-1)([prot_features_emb, mol_features_emb])
        else:
            return Concatenate(axis=-1)([prot_embedding, mol_embedding])

    if model.jointer['name'] == 'combine':
        reg = model.jointer['reg']
        plus1 = model.jointer['plus1']
        nheads = model.jointer['nheads']
        mol_emb_shape = mol_embedding.get_shape().as_list()
        # mol_embedding = (batch_size, emb size)
        # prot_embedding = (batch_size, seq_length, emb size)  # same emb_size

        list_prot_emb = []
        for head in range(nheads):
            att_projection = Dense(mol_emb_shape[1],
                                   kernel_regularizer=regularizers.l2(reg),
                                   bias_regularizer=regularizers.l2(reg))
            # att_projection2 = Dense(mol_emb_shape[1],
            #                        kernel_regularizer=regularizers.l2(reg),
            #                        bias_regularizer=regularizers.l2(reg))

            hmol = att_projection(mol_embedding)
            haa = att_projection(prot_embedding)
            att_w = Lambda(lambda arg: batch_dot_layer(arg[0], arg[1], (2, 1)),
                           name='att_w_' + str(head))([haa, hmol])
            if plus1:
                print('PLUS1')
                att_w = Activation('softmax', name='att_wp_' + str(head))\
                    (Lambda(lambda x: x + 1)(att_w))
            else:
                att_w = Activation('sigmoid', name='att_wp_' + str(head))(att_w)
            list_prot_emb.append(Lambda(lambda arg: batch_dot_layer(arg[0], arg[1], (1, 1)))
                                       ([att_w, haa]))
        prot_emb = Add()(list_prot_emb) if nheads > 1 else list_prot_emb[-1]
        # import pdb; pdb.Pdb().set_trace()
        return Concatenate(axis=-1)([prot_emb, mol_embedding])

    elif model.jointer['name'] == 'combine_linear':
        reg = model.jointer['reg']
        # plus1 = model.jointer['plus1']
        nheads = model.jointer['nheads']

        list_tot_emb = []
        concat_emb = Concatenate(axis=-1)([prot_embedding, mol_embedding])
        concat_emb_shape = concat_emb.get_shape().as_list()
        for head in range(nheads):
            att_w = Dense(concat_emb_shape[1], activation='sigmoid',
                          kernel_regularizer=regularizers.l2(reg),
                          bias_regularizer=regularizers.l2(reg))(concat_emb)
            list_tot_emb.append(Lambda(lambda arg: batch_dot_layer(arg[0], arg[1], (1, 1)))
                                      ([att_w, concat_emb]))

        return Concatenate(axis=-1)(list_tot_emb) if nheads > 1 else list_tot_emb


