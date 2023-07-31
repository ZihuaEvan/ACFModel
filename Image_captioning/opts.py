from __future__ import print_function
import argparse
import misc.utils as utils

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cocobu.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_rela_dir', type=str, default='data/coco_pred_sg',
                        help='path to the directory containing the preprocessed relation matrix')
    parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/cocott_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="transformer",
                    help='transformer')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')


    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip_mode', type=str, default='value',
                    help='value or norm')
    parser.add_argument('--grad_clip_value', type=float, default=0.1,
                    help='clip gradients at this value/max_norm, 0 means no clipping')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_method = greedy, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--max_length', type=int, default=20,
                    help='Maximum length during sampling')
    parser.add_argument('--length_penalty', type=str, default='',
                    help='wu_X or avg_X, X is the alpha')
    parser.add_argument('--block_trigrams', type=int, default=0,
                    help='block repeated trigram.')
    parser.add_argument('--remove_bad_endings', type=int, default=0,
                    help='Remove bad endings')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    # Transformer
    parser.add_argument('--label_smoothing', type=float, default=0,
                    help='')
    parser.add_argument('--noamopt', action='store_true',
                    help='')
    parser.add_argument('--noamopt_rl', type=int, default=0,
                        help='')
    parser.add_argument('--noamopt_warmup', type=int, default=2000,
                    help='')
    parser.add_argument('--noamopt_factor', type=float, default=1,
                    help='')
    parser.add_argument('--reduce_on_plateau', action='store_true',
                    help='')
    parser.add_argument('--reduce_on_plateau_rl', type=int, default=0,
                        help='')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')
    parser.add_argument('--use_warmup', action='store_true',
                    help='warm up the learing rate?')
    # module
    parser.add_argument('--concat_type', type=int, default=0)
    parser.add_argument('--controller', type=int, default=0)
    parser.add_argument('--h_dec', type=int, default=8)

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--rp_decay_every', type=int, default=2500,
                        help='how often to decay learning rate (in iterations)?')
    parser.add_argument('--save_every_epoch', action='store_true',
                    help='Save checkpoint every epoch, will overwrite save_checkpoint_every')
    parser.add_argument('--save_history_ckpt', type=int, default=0,
                    help='If save checkpoints at every save point')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='directory to store checkpointed models')
    parser.add_argument('--cp_id',type=int,default=None ,help='id of checkpoint.')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='9',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')


    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')


    # Structure_loss
    parser.add_argument('--structure_loss_weight', type=float, default=1,
                    help='')
    parser.add_argument('--structure_after', type=int, default=-1,
                    help='T')
    parser.add_argument('--structure_loss_type', type=str, default='seqnll',
                    help='')
    parser.add_argument('--struc_use_logsoftmax', action='store_true', help='')
    parser.add_argument('--entropy_reward_weight', type=float, default=0,
                    help='Entropy reward, seems very interesting')
    parser.add_argument('--self_cider_reward_weight', type=float, default=0,
                    help='self cider reward')

    # Used for self critical or structure. Used when sampling is need during training
    parser.add_argument('--train_sample_n', type=int, default=16,
                    help='The reward weight from cider')
    parser.add_argument('--train_sample_method', type=str, default='sample',
                    help='')
    parser.add_argument('--train_beam_size', type=int, default=1,
                    help='')

    # Used for self critical
    parser.add_argument('--sc_sample_method', type=str, default='greedy',
                    help='')
    parser.add_argument('--sc_beam_size', type=int, default=1,
                    help='')
    parser.add_argument('--drop_worst_after', type=float, default=-1,
                        help='')
    parser.add_argument('--drop_worst_rate', type=float, default=0,
                        help='')
    ################################# Yangxu ####################################

    # gpu id
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_id')

    #rl learning rate
    parser.add_argument('--learning_rate_rl', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_every_rl', type=int, default=5,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate_rl', type=float, default=0.85,
                        help='every how many iterations thereafter to drop LR?(in epoch)')

    # knowledge graph related
    parser.add_argument('--kg_dir', type=str, default='data/kg/kg_info6.npy',
                        help='the path if knowledge graph')
    parser.add_argument('--kg_size', type=int, default=512,
                        help='size of the knowledge graph')
    parser.add_argument('--kge_index', type=str, default='sum',
                        help='how to combine the word embeddings of kg matrix')
    parser.add_argument('--kgfc_index', type=int, default=1,
                        help='which version of kgfc is')
    parser.add_argument('--kglr_decay', type=float, default=10,
                        help='the decay of learning rate of kg related computation')
    parser.add_argument('--kg_mod', type=str, default='matrix',
                        help='matrix: matrix form, embed: embeddings from')
    parser.add_argument('--kg_memory_cell_path', type=str, default='0',
                        help='the path of knowledge graph memory cell')
    parser.add_argument('--kg_fix', type=int, default=0,
                        help='fix knowledge graph dictionary or not')
    parser.add_argument('--mk_batch', type=int, default=0,
                        help='extend mx to batchsize or not')

    # memory of visual features related
    parser.add_argument('--M_size', type=int, default=10000,
                        help='the size of the memory')
    # parser.add_argument('--MX_attr_index', type=int, default=1,
    #                     help='when 1 use one new fc layer, when 0, use the same one as the att feature')
    parser.add_argument('--memory_cell_path', type=str, default='0',
                        help='memory cell path')
    parser.add_argument('--memory_init_decay', type=float, default=1,
                        help='the decay of init memory')
    parser.add_argument('--mx_fix', type=int, default=0,
                        help='fix visual feature dictionary or not')
    parser.add_argument('--mx_batch', type=int, default=0,
                        help='extend mx to batchsize or not')
    parser.add_argument('--km_train', type=int, default=0,
                        help='using k means to learn mx')
    parser.add_argument('--km_alpha', type=float, default=0.98,
                        help='alpha of k means')

    #whether use all previous words when sampling
    parser.add_argument('--all_former', type=int, default=0,
                        help='whether use all previous words when sampling')

    # whether use all previous words when sampling
    parser.add_argument('--val_split', type=str, default='val',
                        help='which set used to val')

    #TTransofrmer
    parser.add_argument('--mask_type', type=int, default=0,
                        help='the type of the mask, when 0, only the next word, when 1, all the next word')
    parser.add_argument('--batch_size_rl', type=int, default=10,
                        help='the batch size when rl')
    parser.add_argument('--ga_index', type=int, default=1,
                        help='when 0, transformer model, when 1, 1-D cluster')
    parser.add_argument('--vis_ga_index', type=int, default=1,
                        help='when 0, transformer model, when 1, 2-D cluster')
    parser.add_argument('--hier_add_index', type=int, default=1,
                        help='when 0, do not add neighbor attention, when add them')
    parser.add_argument('--gelu_index', type=int, default=1,
                        help='whether gelu or not')

    #accumulate batch size
    parser.add_argument('--accumulate_number', type=int, default=1,
                        help='how many times it should accumulate the gradients, the truth batch_size=accumulate_number*batch_size')
    parser.add_argument('--reset_rl_optimzer_index', type=int, default=1,
                        help='reset or not')

    # For diversity evaluation during training
    add_diversity_opts(parser)

    # config
    parser.add_argument('--cfg', type=str, default='acf1.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from misc.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    # default value for start_from and checkpoint_path
    args.checkpoint_path = args.checkpoint_path or './log_%s' %args.id
    # args.start_from = args.start_from or args.checkpoint_path

    # Deal with feature things before anything
    args.use_fc, args.use_att = utils.if_use_feat(args.caption_model)
    if args.use_box: args.att_feat_size = args.att_feat_size + 5

    return args


def add_eval_options(parser):
    # Basic options
    parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
    parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

    # Sampling options
    parser.add_argument('--sample_method', type=str, default='greedy',
                    help='greedy; sample; gumbel; top<int>, top<0-1>')
    parser.add_argument('--beam_size', type=int, default=2,
                    help='indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--max_length', type=int, default=20,
                    help='Maximum length during sampling')
    parser.add_argument('--length_penalty', type=str, default='',
                    help='wu_X or avg_X, X is the alpha')
    parser.add_argument('--group_size', type=int, default=1,
                    help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
    parser.add_argument('--diversity_lambda', type=float, default=0.5,
                    help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_method = sample). Lower = "safer" predictions.')
    parser.add_argument('--decoding_constraint', type=int, default=0,
                    help='If 1, not allowing same word in a row')
    parser.add_argument('--block_trigrams', type=int, default=0,
                    help='block repeated trigram.')
    parser.add_argument('--remove_bad_endings', type=int, default=0,
                    help='Remove bad endings')
    parser.add_argument('--suppress_UNK', type=int, default=1,
                    help='Not predicting UNK')
    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='', 
                    help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--image_root', type=str, default='', 
                    help='In case the image paths have to be preprended with a root path to an image folder')
    # For evaluation on MSCOCO images from some split:
    parser.add_argument('--input_fc_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_att_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_box_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_label_h5', type=str, default='data/cocott_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_json', type=str, default='data/cocott.json',
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add_argument('--split', type=str, default='test', 
                    help='if running on MSCOCO images, which split to use: val|test|train')
    parser.add_argument('--coco_json', type=str, default='', 
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
    # misc
    parser.add_argument('--id', type=str, default='', 
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
    parser.add_argument('--verbose_beam', type=int, default=1, 
                    help='if we need to print out all beam search beams.')
    parser.add_argument('--verbose_loss', type=int, default=0, 
                    help='If calculate loss using ground truth during evaluation')

def add_diversity_opts(parser):
    parser.add_argument('--sample_n', type=int, default=1,
                    help='Diverse sampling')
    parser.add_argument('--sample_n_method', type=str, default='sample',
                    help='sample, bs, dbs, gumbel, topk, dgreedy, dsample, dtopk, dtopp')
    parser.add_argument('--eval_oracle', type=int, default=1, 
                    help='if we need to calculate loss.')


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_opt()