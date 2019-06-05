from keras import backend as K
import keras

TOX21_DATASETS = ['Tox21_' + str(i) for i in range(12)]
MUV_DATASETS = ['MUV_' + str(i) for i in range(17)]
PCBA_DATASETS = ['PCBA_' + str(i) for i in range(90)]

ORIGIN_FOLDER = "./"
LIST_MOL_DATASETS = ['Tox21', 'HIV', 'MUV', 'PCBA', 'PCBA10', 'PCBA100', 'MembranePermeability',
                     'AtomizationEnergy'] + TOX21_DATASETS + MUV_DATASETS + PCBA_DATASETS
LIST_PROT_DATASETS = ['CellLoc', 'SCOPe']
LIST_AA_DATASETS = ['TransmembraneRegions', 'SecondaryStructure', 'MutationStatus']
LIST_DTI_DATASETS = ['DrugBankH', 'DrugBankEC', 'DrugBankHstand', 'DrugBankECstand',
                     "DrugBankHEC-ECstand", "DrugBankHEC-Hstand"]
LIST_MTDTI_DATASETS = ['PCBA_DrugBankH']


LIST_MULTIOUT_DATASETS = ['Tox21', 'MUV', 'PCBA', 'PCBA10', 'PCBA100']
LIST_BINARYCLF_DATASETS = ['Tox21', 'HIV', 'MUV', 'PCBA', 'DrugBankH', 'DrugBankEC'] + \
    ['DrugBankHstand', 'DrugBankECstand', "DrugBankHEC-ECstand", "DrugBankHEC-Hstand"] + \
    TOX21_DATASETS + MUV_DATASETS + PCBA_DATASETS + \
    ['PCBA_DrugBankH', 'PCBA10', 'PCBA100']
LIST_MULTICLF_DATASETS = ['CellLoc', 'SCOPe',
                          'TransmembraneRegions', 'SecondaryStructure', 'MutationStatus']
LIST_CLF_DATASETS = LIST_BINARYCLF_DATASETS + LIST_MULTICLF_DATASETS
LIST_REGR_DATASETS = ['MembranePermeability', 'AtomizationEnergy']
MISSING_DATA_MASK_VALUE = -1


class Dataset():
    def __init__(self, name, n_outputs, loss, final_activation, metrics, ea_metric, ea_mode):
        self.name = name
        self.n_outputs = n_outputs
        self.loss = loss
        self.final_activation = final_activation
        self.metrics = metrics
        self.ea_metric = ea_metric  # earlystopping_metric
        self.ea_mode = ea_mode
        # earlystopping_mode : in min mode, training will stop when the
        # quantity monitored has stopped decreasing; in max mode
        # it will stop when the quantity monitored has stopped increasing;
        # in auto mode, the direction is automatically inferred from the
        # name of the monitored quantity.


def masked_binary_crossentropy(y_true, y_pred, mask_value=MISSING_DATA_MASK_VALUE):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def y_file(dataname, ifold, setting, ratio, foldtype=None):
    if not(setting == 4 or dataname in ["DrugBankHEC-ECstand", "DrugBankHEC-Hstand"]):
        return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' \
            + str(ifold) + '_' + str(setting) + '_' + str(ratio) + '_y.data'
    else:
        ival, ite = ifold
        return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' + foldtype + '_' \
            + str(ifold).replace(' ', '') + '_' + str(setting) + '_' + str(ratio) + '_y.data'


def data_file(dataname, ifold, setting, ratio, foldtype=None):
    if not(setting == 4 or dataname in ["DrugBankHEC-ECstand", "DrugBankHEC-Hstand"]):
        return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' + \
            str(ifold) + '_' + str(setting) + '_' + str(ratio) + '_data.data'
    else:
        return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' + foldtype + '_' \
            + str(ifold).replace(' ', '') + '_' + str(setting) + '_' + str(ratio) + '_data.data'


def MT_y_file(dataname, ival, ite, setting, ratio):
    return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' + \
        str(ival) + ',' + str(ite) + '_' + str(setting) + '_' + str(ratio) + '_y.data'


def MT_data_file(dataname, ival, ite, setting, ratio):
    return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_' + \
        str(ival) + ',' + str(ite) + '_' + str(setting) + '_' + str(ratio) + '_data.data'


def MTextra_y_file(dataname, ival, ite, setting, ratio):
    return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_extra_y.data'


def MTextra_data_file(dataname, ival, ite, setting, ratio):
    return ORIGIN_FOLDER + 'data/' + dataname + '/' + dataname + '_extra_data.data'


def load_dataset_options(dataname):
    if dataname == 'Tox21':
        # max_atoms = 132
        # labels = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
        #           'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        #           'SR-MMP', 'SR-p53']
        # n_samples = 8014
        # n_samples_per_classes = \
        #     [7439, 6902, 6691, 5940, 6316, 7112, 6583, 5935, 7232, 6594, 5920, 6909]
        return Dataset(dataname, 12, masked_binary_crossentropy, 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif 'Tox21' in dataname:
        # Tox21_0 : {0: 7129, 1: 310} ; Tox21_1 : {0: 6664, 1: 238} ;
        # Tox21_2 : {0: 5906, 1: 785} ; Tox21_3 : {0: 5633, 1: 307} ;
        # Tox21_4 : {0: 5518, 1: 798} ; Tox21_5 : {0: 6756, 1: 356} ;
        # Tox21_6 : {0: 6394, 1: 189} ; Tox21_7 : {0: 4974, 1: 961} ;
        # Tox21_8 : {0: 6965, 1: 267} ; Tox21_9 : {0: 6215, 1: 379} ;
        # Tox21_10 : {0: 4984, 1: 936} ; Tox21_11 : {0: 6478, 1: 431}
        return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif dataname == 'HIV':
        # max_atoms = 222
        # labels = ['no', 'yes']
        # n_samples = 41913
        # Counter({0: 40426, 1: 1487})
        return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif dataname == 'MUV':
        # max_atoms = 46
        # labels = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652',
        #           'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737',
        #           'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
        # n_samples = 93127
        # n_samples_per_classes = \
        #     [14844, 14737, 14734, 14633, 14903, 14606, 14647, 14415, 14841, 14691, 14696, 14646,
        #      14676, 14714, 14658, 14775, 14751]
        return Dataset(dataname, 17, masked_binary_crossentropy, 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif 'MUV' in dataname:
        # MUV_0 : {0: 14817, 1: 27} ; MUV_1 : {0: 14708, 1: 29}
        # MUV_2 : {0: 14704, 1: 30} ; MUV_3 : {0: 14603, 1: 30}
        # MUV_4 : {0: 14874, 1: 29} ; MUV_5 : {0: 14577, 1: 29}
        # MUV_6 : {0: 14617, 1: 30} ; MUV_7 : {0: 14387, 1: 28}
        # MUV_8 : {0: 14812, 1: 29} ; MUV_9 : {0: 14663, 1: 28}
        # MUV_10 : {0: 14667, 1: 29} ; MUV_11 : {0: 14617, 1: 29}
        # MUV_12 : {0: 14646, 1: 30} ; MUV_13 : {0: 14684, 1: 30}
        # MUV_14 : {0: 14629, 1: 29} ; MUV_15 : {0: 14746, 1: 29}
        # MUV_16 : {0: 14727, 1: 24}
        return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif dataname == 'MembranePermeability':
        # max_atoms = 59
        # labels = ['ae']
        # n_samples = 201
        return Dataset(dataname, 1, 'mse', None,
                       ['mse'], 'mse', 'min')
    elif dataname == 'AtomizationEnergy':
        # max_atoms = 23
        # labels = ['mp']
        # n_samples = 982
        return Dataset(dataname, 1, 'mse', None,
                       ['mse'], 'mse', 'min')
    elif dataname in ["PCBA", 'PCBA10', 'PCBA100']:
        # max_atoms = 332
        # labels = ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
        #           'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
        #           'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
        #           'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
        #           'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
        #           'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
        #           'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290',
        #           'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314',
        #           'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360',
        #           'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208',
        #           'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339',
        #           'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706',
        #           'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891',
        #           'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453',
        #           'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591',
        #           'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233',
        #           'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170',
        #           'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246',
        #           'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296',
        #           'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644',
        #           'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', ]
        # n_samples = 439863
        return Dataset(dataname, 90, masked_binary_crossentropy, 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
        # return Dataset(dataname, 90, 'categorical_crossentropy', 'sigmoid',
        #                [keras.metrics.binary_accuracy], 'aupr', 'max')

    elif dataname == 'CellLoc':
        # labels = ['chloroplast', 'cytoplasmic', 'ER', 'extracellular', 'Golgi', 'vacuolar',
        #           'lysosomal', 'mitochondrial', 'nuclear', 'peroxisomal', 'plasma_membrane']
        # n_samples = 5917
        # max_seq_length = 5430
        # n_samples_per_class = \
        #     {1: 1392, 10: 1236, 3: 842, 8: 829, 7: 506, 0: 445, 2: 198, 9: 156, 4: 149,
        #      6: 101, 5: 63}
        return Dataset(dataname, 11, 'categorical_crossentropy', 'softmax',
                       [keras.metrics.categorical_accuracy], 'aupr', 'max')
    elif dataname == 'SecondaryStructure':
        # labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
        # n_samples = 4388
        # max_seq_length = 700
        # n_samples_per_class = \
        #     {'H': 313547, 'E': 196870, 'L': 179424, 'T': 104151, 'S': 78118, 'G': 36062,
        #      'B': 9934, 'I': 151}
        return Dataset(dataname, 8, 'categorical_crossentropy', 'softmax',
                       [keras.metrics.categorical_accuracy], 'aupr', 'max')
    elif dataname == 'TransmembraneRegions':
        # labels = ['T', 'S', '.']  # ['T', 'S', '.']
        # n_samples = 16474
        # Counter({'.': 6602073, 't': 508863, 'S': 81576, 'T': 13837})
        # Counter({'.': 6602073, 'T': 522700, 'S': 81576})
        # max_seq_length = 5890
        return Dataset('TransmembraneRegions', 3, 'categorical_crossentropy', 'softmax',
                       [keras.metrics.categorical_accuracy], 'aupr', 'max')
    elif dataname == 'SCOPe':
        # labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        # n_samples = 13963
        # Counter({2: 3976, 3: 3417, 1: 2808, 0: 2520, 6: 706, 4: 273, 5: 263})
        # max_seq_length = 1664
        return Dataset(dataname, 7, 'categorical_crossentropy', 'softmax',
                       [keras.metrics.categorical_accuracy], 'aupr', 'max')
    # elif dataname == 'MutationStatus':
    #     # list_labels = ['Polymorphism', 'Disease']
    #     return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
    #                    [mcc, aupr, auroc, f1], 'aupr', 'max')

    elif 'DrugBankH' in dataname:
        # list_labels = ['no', 'yes']
        # n_pos_samples = 13265
        # max_atoms = 457
        # max_seq_length = 5179
        return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')
    elif 'DrugBankEC' in dataname:
        # list_labels = ['no', 'yes']
        # n_pos_samples = 876
        # max_atoms = 112
        # max_seq_length = 1407
        return Dataset(dataname, 1, 'binary_crossentropy', 'sigmoid',
                       [keras.metrics.binary_accuracy], 'aupr', 'max')

    elif 'PCBA_DrugBankH' in dataname:
        # list_labels = ['no', 'yes']
        # n_pos_samples = 13265
        # max_seq_length = 5179
        return Dataset(dataname, [1, 91], [masked_binary_crossentropy, masked_binary_crossentropy],
                       ['sigmoid', 'sigmoid'],
                       [keras.metrics.binary_accuracy, keras.metrics.binary_accuracy],
                       ['aupr', 'aupr'], ['max', 'max'])
