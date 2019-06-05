from src.experiments.mol_experiments import *


if __name__ == "__main__":
    if sys.argv[2] in LIST_MOL_DATASETS:
        exp = MolExperiment('essai')
    elif sys.argv[2] in LIST_PROT_DATASETS or sys.argv[2] in LIST_AA_DATASETS:
        exp = ProtExperiment('essai')
    elif sys.argv[2] in LIST_DTI_DATASETS:
        exp = DTIExperiment('essai')
    elif sys.argv[2] in LIST_MTDTI_DATASETS:
        exp = MTDTIExperiment('essai')

    exp._get_parameters()

    model_args = exp.args.batch_size, exp.args.n_epochs, exp.args.init_lr, \
        exp.args.patience_early_stopping, \
        exp.args.lr_scheduler, exp.args.pred_layers, exp.args.pred_batchnorm, \
        exp.args.pred_dropout, exp.args.pred_reg, exp.args.cpu_workers_for_generator, \
        exp.args.queue_size_for_generator, \
        exp.dataset, \
        exp.enc_dict_param

    if exp.dataset.name in LIST_MOL_DATASETS:
        exp.model = MolModel(*model_args)
    elif exp.dataset.name in LIST_PROT_DATASETS or exp.dataset.name in LIST_AA_DATASETS:
        exp.model = ProtModel(*model_args)
    elif exp.dataset.name in LIST_DTI_DATASETS:
        exp.model = DTIModel(*model_args)
    elif exp.dataset.name in LIST_DTI_DATASETS:
        exp.model = MTDTIModel(*model_args)

    print(str(exp.model))
