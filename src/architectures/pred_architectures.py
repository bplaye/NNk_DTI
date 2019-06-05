from keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, Activation
# from keras import regularizers


def FNN_pred(model, emb, name_prefix=''):
    # print('EMB', emb)
    x = emb
    for n in range(len(model.pred_layers)):
        x = Dense(model.pred_layers[n], activation=None,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  # kernel_regularizer=regularizers.l2(model.pred_reg),
                  # bias_regularizer=regularizers.l2(model.pred_reg),
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                  name=name_prefix + 'pred_layer_' + str(n))(x)
        if model.pred_reg != 0:
            x = GaussianNoise(model.pred_reg)(x)
        x = Activation('relu')(x)
        if model.pred_dropout != 0.:
            x = Dropout(model.pred_dropout, noise_shape=None, seed=None)(x)
        else:
            print('drop_pred is 0 ###########')
        if model.pred_BN is True:
            x = BatchNormalization(axis=-1, name=name_prefix + 'BNpred_layer_' + str(n))(x)
            # emb must be (batch_size, emb_size)
        else:
            print("no pred BN ########")
    pred = Dense(model.dataset.n_outputs, activation=model.dataset.final_activation,
                 name=name_prefix + 'output')(x)
    # print('pred', pred)
    return pred
