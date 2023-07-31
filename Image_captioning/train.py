from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import opts
import models
from dataloader import *
#import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    transformer_based_models=['transformer',  'ACF']

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.word_mask_ix = loader.dataset.mask_index
    opt.word_end_ix = loader.dataset.end_index

    ##########################
    # Initialize infos, histories, and tensorboard
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }

    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + str(opt.cp_id).zfill(4) +'.pkl')):
        print("Load from checkpoint_id {0}".format(opt.cp_id))
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + str(opt.cp_id).zfill(4) +'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme

    elif opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    histories = defaultdict(dict)

    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path, 'histories_' + opt.id+ str(opt.cp_id).zfill(4) + '.pkl')):
        with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id+ str(opt.cp_id).zfill(4) + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))
        print('start from checkpoint is {0}'.format(opt.checkpoint_path + str(opt.cp_id)))

    elif opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))
        print('start from is {0}'.format(opt.start_from))

    else:
        print("history is not found")
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab

    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path,"model"+ opt.id+ str(opt.cp_id).zfill(4) + '.pth')):
        print("Load success from checkpoint")
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,"model"+ opt.id+ str(opt.cp_id).zfill(4) + '.pth')))

    elif opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        print("Load success from latest")
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel
    dp_model = torch.nn.DataParallel(model)
    dp_model.vocab = getattr(model, 'vocab', None)
    dp_lw_model = torch.nn.DataParallel(lw_model)
    # dp_model = model
    # dp_lw_model = lw_model

    ##########################
    #  if begin with rl, then use rl parameter to initialize the optimizer
    ##########################
    iteration = infos['iter']
    epoch = infos['epoch']
    if opt.self_critical_after != -1 or opt.structure_after != -1 and epoch >= opt.self_critical_after and epoch >=  opt.structure_after:
        opt.noamopt = opt.noamopt_rl
        opt.reduce_on_plateau = opt.reduce_on_plateau_rl
    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.caption_model in transformer_based_models, \
            'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        print("naomopt or reduce_on_plateau is nessary")
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path, "optimizer"+ opt.id+ str(opt.cp_id).zfill(4) + '.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(os.path.join(opt.checkpoint_path, "optimizer"+ opt.id+ str(opt.cp_id).zfill(4) + '.pth'))))

    elif opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    #iteration = infos['iter']
    #epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    if opt.noamopt:
        optimizer._step = iteration
    epoch_done = True
    #rroi = opt.reset_rl_optimzer_index
    reset_rl_optimzer_index = True
    # Assure in training mode

    dp_lw_model.train()
    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                if opt.structure_after != -1 and epoch >= opt.structure_after and reset_rl_optimzer_index:
                    opt.learning_rate_decay_start = opt.self_critical_after
                    opt.learning_rate_decay_rate = opt.learning_rate_decay_rate_rl
                    opt.learning_rate_decay_every = opt.learning_rate_decay_every_rl
                    opt.learning_rate = opt.learning_rate_rl
                    opt.noamopt = opt.noamopt_rl
                    opt.reduce_on_plateau = opt.reduce_on_plateau_rl
                    if opt.noamopt:
                        assert opt.caption_model in transformer_based_models, \
                            'noamopt can only work with transformer'
                        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
                    elif opt.reduce_on_plateau:
                        optimizer = utils.build_optimizer(model.parameters(), opt)
                        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                    else:
                        optimizer = utils.build_optimizer(model.parameters(), opt)
                    reset_rl_optimzer_index = False

                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False

                # If start structure loss training
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False
                if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                    drop_worst_flag = True
                else:
                    drop_worst_flag = False

                epoch_done = False

            start = time.time()
            # Load data from train split (0)
            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['att_masks'], data['train_labels'],
                   data['test_labels'], data['train_masks'], data['test_masks'], data['batch_index']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, train_labels, test_labels, \
            train_masks, test_masks, batch_index = tmp

            # for gt in data['gts']:
            #     gt[gt == 0] = opt.word_end_ix
            optimizer.zero_grad()
            # batch_index = batch_index.squeeze()
            model_out = dp_lw_model(fc_feats, att_feats, train_labels, train_masks, att_masks, data['gts'],
                        torch.arange(0, len(data['gts'])), sc_flag, struc_flag, test_masks=test_masks,
                                    test_labels=test_labels, batch_index = batch_index)

            if not drop_worst_flag:
                loss = model_out['loss'].mean()
            else:
                loss = model_out['loss']
                loss = torch.topk(loss, k=int(loss.shape[0] * (1 - opt.drop_worst_rate)), largest=False)[0].mean()

            begin_time = time.time()
            loss.backward()
            end_time = time.time()
            print("backward time:", end_time - begin_time)

            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()
            if struc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()
            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                save_id = iteration / opt.save_checkpoint_every
                utils.save_checkpoint_new(opt, model, infos, optimizer, save_id, histories)
            if iteration % opt.rp_decay_every == 0:
                # eval model
                eval_kwargs = {'split': opt.val_split,
                               # 'num_images': 10,
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)


                #optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score





    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)#, histories)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# torch.cuda.set_device(opt.gpu)

train(opt)
