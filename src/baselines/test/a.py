def DrugBank_CV(DB, list_ratio, setting, cluster_cv, n_folds=5, seed=324):
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)
    mratio = max(list_ratio)
    list_ratio = np.sort(list_ratio)[::-1].tolist()
    print('DB got')
    np.random.seed(seed)

    if setting == 1:
        ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
        n_folds = 5
        np.random.seed(seed)

        # pos folds
        pos_folds_data, pos_folds_y = [], []
        list_couple, y = [], []
        for i in range(len(ind_inter[0])):
            list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
            y.append(1)
        y, list_couple = np.array(y), np.array(list_couple)
        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        ifold = 0
        for train_index, test_index in skf.split(X):
            print(len(train_index), len(test_index))
            pos_folds_data.append(list_couple[test_index].tolist())
            pos_folds_y.append(y[test_index].tolist())
            ifold += 1
        for n in range(n_folds):
            for n2 in range(n_folds):
                if n2 != n:
                    for c in pos_folds_data[n2]:
                        if c in pos_folds_data[n]:
                            print(c)
                            exit(1)

        # neg folds
        neg_folds_data, neg_folds_y = {r: [] for r in list_ratio}, {r: [] for r in list_ratio}
        mmask = np.random.choice(np.arange(len(ind_non_inter[0])), len(ind_inter[0]) * mratio,
                                 replace=False)
        ind_non_inter = (ind_non_inter[0][mmask], ind_non_inter[1][mmask])

        list_couple, y = [], []
        for i in range(len(ind_non_inter[0])):
            list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
                                dict_ind2mol[ind_non_inter[1][i]]))
            y.append(0)
        list_couple, y = np.array(list_couple), np.array(y)
        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        ifold = 0
        for train_index, test_index in skf.split(X):
            neg_folds_data[mratio].append(np.array(list_couple)[test_index].tolist())
            neg_folds_y[mratio].append(np.array(y)[test_index].tolist())
            ifold += 1

        previous_nb_non_inter = len(ind_inter[0]) * mratio
        for ir, ratio in enumerate(list_ratio):
            print(ratio)
            if ratio != mratio:
                nb_non_inter = \
                    round((float(ratio) / (float(list_ratio[ir - 1]))) *
                          previous_nb_non_inter)
                previous_nb_non_inter = nb_non_inter
                nb_non_inter = round(float(nb_non_inter) / float(n_folds))
                print('nb_non_inter', previous_nb_non_inter)
                print(len(neg_folds_data[list_ratio[ir - 1]][0]))
                for ifold in range(n_folds):
                    mask = np.random.choice(
                        np.arange(len(neg_folds_data[list_ratio[ir - 1]][ifold])),
                        nb_non_inter, replace=False)
                    neg_folds_data[ratio].append(
                        np.array(neg_folds_data[list_ratio[ir - 1]][ifold])[mask].tolist())
                    neg_folds_y[ratio].append(
                        np.array(neg_folds_y[list_ratio[ir - 1]][ifold])[mask].tolist())
                    ifold += 1
            print('nb_non_inter', previous_nb_non_inter)

        # save folds
        for ir, ratio in enumerate(list_ratio):
            print("ratio", ratio)
            fo = open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
                      '_folds.txt', 'w')
            folds_data = []
            for ifold in range(n_folds):
                datatemp = pos_folds_data[ifold] + neg_folds_data[ratio][ifold]
                folds_data.append([c[0] + c[1] for c in datatemp])
                ytemp = pos_folds_y[ifold] + neg_folds_y[ratio][ifold]
                # import pdb; pdb.Pdb().set_trace()
                fo.write("ifold " + str(ifold) + '\t' + str(collections.Counter(ytemp)) + '\n')
                pickle.dump(datatemp, open(data_file(DB, ifold, setting, ratio), 'wb'))
                pickle.dump(ytemp, open(y_file(DB, ifold, setting, ratio), 'wb'))
                print("ifold " + str(ifold), str(collections.Counter(ytemp)))
            fo.close()
            for n in range(n_folds):
                for n2 in range(n_folds):
                    if n2 != n:
                        for c in folds_data[n2]:
                            if c in folds_data[n]:
                                print('alerte', c)
                                exit(1)
