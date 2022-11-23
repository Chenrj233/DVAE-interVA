import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim
import math
from data import MonoTextData
from modules import VAE
from modules import GaussianLSTMEncoder, LSTMOurDecoder

from exp_utils import create_exp_dir
from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, sample_sentences, \
    visualize_latent, reconstruct
from itertools import product
from tqdm import trange, tqdm


clip_grad = 5.0
decay_epoch = 5
lr_decay = 0.5
max_decay = 5
lr_warm_up_epoch = 0
lr_start = 0

logging = None


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--att_gamma', type=float, default=0.0)
    parser.add_argument('--cycle', type=int, default=0)
    # dataset
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                        help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # decoding
    parser.add_argument('--sample_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--sample', type=int, default=0, help="the number of sample text")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10,
                        help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=0.0, help="starting KL weight")

    parser.add_argument('--kl_att', type=float, default=1.0, help="attention KL weight")


    parser.add_argument('--adv_weight', type=float, default=1.0, help="adv weight")
    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                        help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=500,
                        help="save checkpoint every epoch before this number")

    parser.add_argument("--fix_var", type=float, default=-1)
    parser.add_argument("--fix_att_var", type=float, default=-1)
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1.)


    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    load_str = "_load" if args.load_path != "" else ""

    # set load and save paths

    if args.exp_dir == None:
        args.exp_dir = "exp_{}{}/{}_warm{}_kls{:.1f}_disdim{}_" \
                       "attweight{}_adv{}_attgamma{:.4f}_gamma{}".format(args.dataset,
                                                                         load_str, args.dataset,
                                                                         args.warm_up,
                                                                         args.kl_start,
                                                                         args.disen_dim,
                                                                         args.kl_att,
                                                                         args.adv_weight,
                                                                         args.att_gamma,
                                                                         args.gamma)

    if len(args.load_path) <= 0 and args.eval:
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args


def sample_from_prior(model, z, y, strategy):
    assert z.size(0) == y.size(0)

    decoded_batch = model.decode(z, y, strategy=strategy)
    text = []
    for i in range(len(decoded_batch)):
        text.append(" ".join(decoded_batch[i][:-1]))

    return text

def sample_from_prior_save(model, z, y, strategy):
    assert z.size(0) == y.size(0)
    decoded_batch, save_data = model.decode(z, y, strategy=strategy, save_vector=True)
    '''for item_name, item in save_data.items():
        print(item_name, item)
    exit()
    text = []'''

    return save_data




def test(model, test_data_batch, test_label_batch, mode, args, kl_weight, kl_att, adv_weight, verbose=True):
    global logging

    report_kl_loss = report_rec_loss = report_loss = report_att_kl_loss = \
        report_adv = report_dis = report_emb = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_label = test_label_batch[i]
        batch_label = torch.tensor(batch_label, dtype=torch.float32,
                                   requires_grad=False, device=args.device)
        batch_size, sent_len = batch_data.size()
        batch_label = batch_label.view(batch_size, 1)

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size

        loss, loss_rc, loss_kl, loss_att_KL, adv_loss, dis_loss = model.loss_att(batch_data, batch_label,
                                                                                        kl_weight, kl_att,
                                                                                              adv_weight,
                                                                                        nsamples=args.nsamples)

        assert (not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss_att_KL = loss_att_KL.sum()

        loss_adv = adv_loss.sum()
        loss_dis = dis_loss.sum()


        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_att_kl_loss += loss_att_KL.item()


        report_adv += loss_adv
        report_dis += loss_dis
        if args.warm_up == 0 and args.kl_start < 1e-6:
            report_loss += loss_rc.item()
        else:
            report_loss += loss.item()


    test_loss = report_loss / report_num_sents

    kl = report_kl_loss / report_num_sents
    att_kl = report_att_kl_loss / report_num_sents
    rec = report_rec_loss / report_num_sents
    adv = report_adv / report_num_sents
    dis = report_dis / report_num_sents

    if verbose:
        logging('%s --- avg_loss: %.4f, kl: %.4f, att_kl: %.4f, rec: %.4f, adv: %.4f, dis: %.4f,' % \
                (mode, test_loss, kl, att_kl, rec, adv, dis))

    return test_loss, kl, att_kl, rec, adv, dis


def main(args):
    global logging
    debug = (args.sample_from != "" or args.eval == True) #don't make exp dir for reconstruction
    logging = create_exp_dir(args.exp_dir, scripts_to_save=None, debug=debug)

    if args.cuda:
        logging('using cuda')
    logging(str(args))


    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    device = "cuda" if args.cuda else "cpu"
    args.device = device

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)


    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)
    tol_train = len(train_data)
    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)

    log_niter = (len(train_data) // args.batch_size) // 10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)


    if args.enc_type == 'lstm':
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMOurDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(device)


    if args.load_path:
        loaded_state_dict = torch.load(args.load_path, map_location=args.device)
        vae.load_state_dict(loaded_state_dict, strict=False)
        logging("%s loaded" % args.load_path)


        if args.reset_dec:
            if args.gamma > 0:
                vae.encoder.reset_parameters(model_init, emb_init, reset=True)
            logging("\n-------reset decoder-------\n")
            vae.decoder.reset_parameters(model_init, emb_init)


    if args.eval:
        logging('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test(vae, test_data_batch, test_label_batch, "TEST", args)

        return

    if args.sample_from != "":
        print('begin decoding')
        if args.load_path == "":
            vae.load_state_dict(torch.load(args.sample_from, map_location=args.device))
        vae.eval()
        save_dir = "samples/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with torch.no_grad():
            c_max = 0.90
            c_min = 0.05
            per_qu = (c_max - c_min) / (args.sample - 1)
            c_list = [c_min + i * per_qu for i in range(args.sample)]
            all_text = []
            all_label = []
            sample_time = 10
            for i in trange(args.sample):
                z = vae.sample_from_prior(10)
                label = [c_list[i] for j in range(sample_time)]
                y = torch.tensor([c_list[i]] * sample_time, dtype=torch.float32, device=args.device)
                y = y.view(z.size(0), 1)
                text = sample_from_prior(vae, z, y, args.decoding_strategy)
                all_text += text
                all_label += label

            assert len(all_label) == len(all_text)
            import pandas as pd
            out_df = pd.DataFrame({"text": all_text, "label": all_label})
            out_df.to_csv(os.path.join(save_dir, args.exp_dir.split("/")[-1] + "_{}.csv".format(args.decoding_strategy)), header=None, index=None)
        return




    if args.opt == "sgd":
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, momentum=args.momentum)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=args.lr, momentum=args.momentum)
        adv_optimizer = optim.SGD(vae.adv.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001)
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001)
        adv_optimizer = optim.Adam(vae.adv.parameters(), lr=0.001)
        opt_dict['lr'] = 0.001
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    kl_att_weight = args.kl_att


    if args.warm_up > 0:
        anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    else:
        anneal_rate = 0


    train_data_batch, train_label_batch = train_data.create_data_batch_labels(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)

    print("Total iter: %d" %len(train_data_batch))
    if lr_warm_up_epoch == 0:
        opt_dict['lr'] = args.lr
    else:
        lr_per_iter = (args.lr - lr_start) / len(train_data_batch) / lr_warm_up_epoch
        opt_dict['lr'] = lr_start
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in trange(args.epochs):
            vae.train()
            tol_loss = tol_rec = tol_kl = tol_att_kl = tol_adv = tol_dis_mae = tol_clf_adv = tol_emb = 0
            report_kl_loss = report_rec_loss = report_loss = report_att_kl_loss = \
                report_adv = report_dis_mae = report_clf_adv = 0
            report_num_words = report_num_sents = 0

            if args.cycle > 0 and (epoch - 1) % args.cycle == 0:
                kl_weight = args.kl_start
                print('KL Annealing restart!')
            for i in np.random.permutation(len(train_data_batch)):

                batch_data = train_data_batch[i]
                batch_label = train_label_batch[i]
                batch_label = torch.tensor(batch_label, dtype=torch.float32,
                                        requires_grad=False, device=device)

                batch_size, sent_len = batch_data.size()
                batch_label = batch_label.view(batch_size, 1)


                if batch_data.size(0) < 2:
                    continue

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                kl_weight = min(1.0, kl_weight + anneal_rate)



                adv_optimizer.zero_grad()
                clf_adv = vae.loss_att_adv(batch_data, batch_label, nsamples=args.nsamples)
                loss_clf_adv = clf_adv.sum()
                clf_adv = clf_adv.mean(dim=-1)
                clf_adv.backward()
                adv_optimizer.step()

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                loss, loss_rc, loss_kl, loss_att_KL, adv_loss, dis_mae_loss = \
                    vae.loss_att(batch_data, batch_label, kl_weight, kl_att_weight,  args.adv_weight,
                                            nsamples=args.nsamples)


                report_loss += loss.sum().item()
                tol_loss += loss.sum().item()

                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()
                loss_att_KL = loss_att_KL.sum()
                loss_dis_mae = dis_mae_loss.sum()
                loss_adv = adv_loss.sum()

                enc_optimizer.step()
                dec_optimizer.step()

                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()
                report_att_kl_loss += loss_att_KL.item()
                report_adv += loss_adv.item()
                report_dis_mae += loss_dis_mae.item()
                report_clf_adv += loss_clf_adv.item()



                tol_kl += loss_kl.item()
                tol_att_kl += loss_att_KL.item()
                tol_rec += loss_rc.item()
                tol_adv += loss_adv.item()
                tol_dis_mae += loss_dis_mae.item()
                tol_clf_adv += loss_clf_adv.item()


                if iter_ % log_niter == 0:
                    train_loss = report_loss / report_num_sents

                    logging('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, att_kl: %.4f, recon: %.4f, ' \
                            'adv: %.4f, dis_mae: %.4f, clf: %.4f'
                            ' time %.2fs, kl_weight: %.4f, kl_att_weight: % .4f' %
                            (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                             report_att_kl_loss / report_num_sents,
                             report_rec_loss / report_num_sents,
                              report_adv / report_num_sents, report_dis_mae / report_num_sents,
                             report_clf_adv / report_num_sents,
                             time.time() - start, kl_weight, kl_att_weight))

                    # sys.stdout.flush()

                    report_rec_loss = report_kl_loss = report_loss = \
                        report_att_kl_loss = report_adv = report_dis_mae = report_clf_adv = report_emb = 0
                    report_num_words = report_num_sents = 0

                iter_ += 1
                if lr_warm_up_epoch != 0 and iter_ <= lr_warm_up_epoch * len(train_data_batch):
                    opt_dict['lr'] = lr_per_iter * iter_

            logging('kl weight %.4f' % kl_weight)
            logging('lr {}'.format(opt_dict["lr"]))

            vae.eval()
            with torch.no_grad():
                if epoch+1 <= args.warm_up:
                    test_loss, kl, att_kl, rec, adv, dis_mae = test(vae, test_data_batch,
                                                                                       test_label_batch, "VAL", args,
                                                                                       1.0, 1.0,
                                                                                    args.adv_weight)
                else:
                    test_loss, kl, att_kl, rec, adv, dis_mae = test(vae, test_data_batch,
                                                                                       test_label_batch, "VAL", args,
                                                                                       kl_weight, kl_att_weight,
                                                                                       args.adv_weight)
                logging('Val   loss: avg_loss: %.4f, kl: %.4f, att_kl: %.4f, rec: %.4f, adv: %.4f, dis_mae: %.4f,' % \
                        (test_loss, kl, att_kl, rec, adv, dis_mae))


            if args.save_ckpt > 0 and epoch <= args.save_ckpt and (epoch + 1) % 20 == 0:
                logging('save checkpoint')
                torch.save(vae.state_dict(), os.path.join(args.exp_dir, f'model_ckpt_{epoch}.pt'))

            tol_loss = tol_loss / tol_train
            tol_kl = tol_kl / tol_train
            tol_att_kl = tol_att_kl / tol_train
            tol_rec = tol_rec / tol_train
            tol_adv = tol_adv / tol_train
            tol_dis_mae = tol_dis_mae / tol_train
            tol_clf_adv = tol_clf_adv / tol_train
            logging('Train loss: avg_loss: %.4f, kl: %.4f, att_kl: %.4f, rec: %.4f, adv: %.4f, '
                    'dis_mae: %.4f, clf: %.4f,' %
                    (tol_loss, tol_kl, tol_att_kl, tol_rec,
                                                     tol_adv, tol_dis_mae, tol_clf_adv))

            if test_loss < best_loss:
                logging('update best loss')
                best_loss = test_loss
                best_kl = kl
                best_att_kl = att_kl
                best_rec = rec
                best_adv = adv
                best_dis_mae = dis_mae

                logging('Best val loss: tol loss: %.4f, kl: %.4f, att_kl: %.4f, recon: %.4f, ' \
                            'adv: %.4f, dis_mae: %.4f,' % (best_loss, best_kl, best_att_kl, best_rec,
                        best_adv, best_dis_mae))
                torch.save(vae.state_dict(), args.save_path)

            if test_loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch:
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    vae.load_state_dict(torch.load(args.save_path))
                    logging('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    adv_optimizer = optim.SGD(vae.adv.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = test_loss

            if decay_cnt == max_decay:
                break





    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')





if __name__ == '__main__':

    args = init_config()
    torch.cuda.set_device(args.gpu)

    load_str = "_load" if args.load_path != "" else ""

    args.save_path = os.path.join(args.exp_dir, 'model.pt')
    if args.sample != 0:
        args.sample_from = args.save_path

    main(args)
