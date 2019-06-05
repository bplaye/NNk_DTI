import pickle
import collections
from src.utils.prot_utils import LIST_AA
import numpy as np


def process_DB():
    DB = 'MutationStatus'

    path = 'data/MutationStatus/'

    list_labels = ['Polymorphism', 'Disease']

    dict_ShortName2Letter = {'Val': 'V', 'Thr': 'T', 'Lys': 'K', 'Trp': 'W', 'Arg': 'R',
                             'His': 'H', 'Asn': 'N', 'Ser': 'S', 'Met': 'M', 'Ala': 'A',
                             'Phe': 'F', 'Leu': 'L', 'Pro': 'P', 'Asp': 'D', 'Cys': 'C',
                             'Glu': 'E', 'Gln': 'Q', 'Ile': 'I', 'Gly': 'G', 'Tyr': 'Y',
                             'Unknown': 'X'
                             }
    # 'Asx': 'B', 'Glx': 'Z'

    dict_mut_per_gene = {}
    dict_fasta_per_prot = {}
    fin = open(path +
               'uniprot-human-filtered-reviewed%3Ayes+AND+' +
               'organism%3A%22Homo+sapiens+%28Human%29--.fasta', 'r')
    uniprot_id = None
    for line in fin:
        line = line.rstrip()
        if line[0] == '>':
            if uniprot_id is not None:
                dict_fasta_per_prot[uniprot_id] = fasta
            uniprot_id = line.split('|')[1]
            fasta = ''
        else:
            fasta += line
    dict_fasta_per_prot[uniprot_id] = fasta
    fin.close()

    list_gene, LIST_AA_rejected, list_length_rejected = [], [], []
    fin = open(path + 'humsavar.txt', 'r')
    for line in fin:
        line = line.rstrip()
        list_line = line.split(' ')
        gene = list_line[0]
        for el in list_line:
            if 'uniprot' in el:
                uniprot_id = line.split('"')[1].split('/')[-1]
            elif 'expasy' in el:
                var_id = line.split('"')[1].split('/')[-1][:-5]
            elif el in ['Polymorphism', 'Disease', 'Unclassified']:
                label = el
            elif el[:2] == 'p.':
                aa0, aa1, pos = el[2:5], el[-3:], int(el[5:-3]) - 1
        if uniprot_id in dict_fasta_per_prot.keys():
            fasta = dict_fasta_per_prot[uniprot_id]

            issue = False
            for aa in fasta:
                if aa not in LIST_AA:
                    issue = True
                    LIST_AA_rejected.append(var_id)
                    break
            if issue is False and len(fasta) > 1000:
                list_length_rejected.append(var_id)
                issue = True

            if issue is False:
                list_gene.append(var_id)
                if fasta[pos] != dict_ShortName2Letter[aa0]:
                    print(fasta)
                    print(fasta[pos - 2: pos + 3])
                    print(fasta[pos])
                    print(dict_ShortName2Letter[aa0])
                    print(dict_ShortName2Letter[aa1])
                    print('error')
                    exit(1)
                else:
                    fasta = fasta[:pos] + dict_ShortName2Letter[aa1] + fasta[pos + 1:]
                if gene not in dict_mut_per_gene.keys():
                    dict_mut_per_gene[gene] = {lab: [] for lab in ['Polymorphism', 'Disease',
                                                                   'Unclassified']}
                dict_mut_per_gene[gene][label].append((var_id, fasta, pos))
            else:
                print(uniprot_id + ' doesnt have fasta')

        uniprot_id, label, var_id, aa0, aa1, pos = None, None, None, None, None, None
    fin.close()
    print('number of proteins in the dataset:', len(list_gene))
    print('number of rejected proteins because unknown aa:', len(LIST_AA_rejected))
    print('number of rejected proteins because too long:', len(list_length_rejected))

    list_X, labels, list_nb, list_nb_pol, list_nb_dis, list_seq_length = [], [], [], [], [], []
    max_seq_length, list_fasta, list_prot, list_label = 0, [], [], []
    labels_for_fold = []
    # fout = open(path + 'mutation_status.fasta', 'w')
    for gene in list(dict_mut_per_gene.keys()):
        nb = min(len(dict_mut_per_gene[gene]['Polymorphism']),
                 len(dict_mut_per_gene[gene]['Disease']))
        list_nb.append(nb)
        list_nb_pol.append(len(dict_mut_per_gene[gene]['Polymorphism']))
        list_nb_dis.append(len(dict_mut_per_gene[gene]['Disease']))

        # if nb < len(dict_mut_per_gene[gene]['Polymorphism']):
        #   pol = np.random.choice(dict_mut_per_gene[gene]['Polymorphism'], size=nb, replace=False)
        # else:
        #   pol = dict_mut_per_gene[gene]['Polymorphism']
        # if nb < len(dict_mut_per_gene[gene]['Disease']):
        #   dis = np.random.choice(dict_mut_per_gene[gene]['Disease'], size=nb, replace=False)
        # else:
        #   dis = dict_mut_per_gene[gene]['Disease']

        pol, dis = dict_mut_per_gene[gene]['Polymorphism'], dict_mut_per_gene[gene]['Disease']
        import pdb; pdb.Pdb().set_trace()

        for i in range(len(pol)):
            var_id, fasta, pos = pol[i]
            # X_local = np.zeros((1, max_seq_length, len(LIST_AA), 1), dtype=np.int32)
            # iaa=0
            # for aa in fasta:
            #     X_local[0, iaa, dict_aa[aa], 0] = 1
            #     iaa+=1

            # label_local = np.zeros((1, max_seq_length), dtype=np.int32)
            # label_local[0, pos] = 1
            # labels.append(label_local)
            # labels_for_fold.append(1)
            # list_X.append(X_local)
            list_prot.append()
            list_label.append()
            list_seq_length.append(len(fasta))
            if len(fasta) > max_seq_length:
                max_seq_length = len(fasta)
            # fout.write('>' + var_id + '\n' + fasta + '\n')

        for i in range(len(dis)):
            var_id, fasta, pos = dis[i]
            # X_local = np.zeros((1, max_seq_length, len(LIST_AA), 1), dtype=np.int32)
            # iaa = 0
            # for aa in fasta:
            #     X_local[0, iaa, dict_aa[aa], 0] = 1
            #     iaa += 1

            # label_local = np.zeros((1, max_seq_length), dtype=np.int32)
            # label_local[0, pos] = 2
            # labels.append(label_local)
            # labels_for_fold.append(2)
            # list_X.append(X_local)
            list_seq_length.append(len(fasta))
            if len(fasta) > max_seq_length:
                max_seq_length = len(fasta)
            # fout.write('>' + var_id + '\n' + fasta + '\n')

    # fout.close()
    print(np.sum(list_nb))
    print(np.sum(list_nb_pol))
    print(np.sum(list_nb_dis))

    # X = np.concatenate(list_X, axis=0)
    # np.save(path+'MutationStatus_X', X)

    list_ID, list_FASTA, list_y, dict_id2fasta = list_prot, list_fasta, list_label, {}
    for ip in range(len(list_prot)):
        dict_id2fasta[list_prot[ip]] = list_FASTA[ip]
        if len(list_FASTA[ip]) != len(list_y[ip]):
            print(ip, len(list_FASTA[ip]), len(list_y[ip]))

    pickle.dump(dict_id2fasta,
                open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    # pickle.dump(dict_uniprot2fasta,
    #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    pickle.dump(list_FASTA,
                open('data/' + DB + '/' + DB + '_list_FASTA.data', 'wb'))
    pickle.dump(list_y,
                open('data/' + DB + '/' + DB + '_list_y.data', 'wb'))
    pickle.dump(list_ID,
                open('data/' + DB + '/' + DB + '_list_ID.data', 'wb'))

    f = open('data/' + DB + '/' + DB + '_dict_ID2FASTA.tsv', 'w')
    for cle, valeur in dict_id2fasta.items():
        f.write(cle + '\t' + valeur + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_FASTA.tsv', 'w')
    for s in list_FASTA:
        f.write(s + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_y.tsv', 'w')
    for s in list_y:
        f.write(str(s) + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_ID.tsv', 'w')
    for s in list_ID:
        f.write(s + '\n')
    f.close()

    print(len(list_FASTA))
    print(collections.Counter([el for y in list_y for el in y]))
    print('max_seq_length', max_seq_length)


if __name__ == "__main__":
    process_DB()

