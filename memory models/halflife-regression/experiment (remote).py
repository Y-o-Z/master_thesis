"""
Copyright (c) 2016 Duolingo Inc. MIT Licence.  [adapted @ Yoel Zweig]

Python script that implements spaced repetition models from Settles & Meeder (2016).
Recommended to run with pypy for efficiency. See README.
"""

import argparse
import csv
import gzip
import math
import os
import random
import sys

from collections import defaultdict, namedtuple


# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2.)


# data instance object
Instance = namedtuple('Instance', 'p t fv h a lang right wrong ts uid lexeme'.split())


class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements the following approaches:
      - 'hlr' (half-life regression; trainable)
      - 'lr' (logistic regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
      - 'my_pimsleur' (fixed)
      - 'constant' (fixed)
    """
    def __init__(self, method='hlr', omit_h_term=False, initial_weights=None, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma

    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.):
        if self.method == 'hlr':
            h = self.halflife(inst, base)
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'leitner':
            try:
                h = hclip(2. ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'pimsleur' or self.method == "my_pimsleur":
            if self.method == "pimsleur":
                try:
                    h = hclip(2. ** (2.35*inst.fv[0][1] - 16.46))
                except OverflowError:
                    h = MAX_HALF_LIFE
            else:
                try:
                    h = hclip(2. ** (math.log(5, 2)*inst.fv[0][1]))
                except OverflowError:
                    h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'lr':
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            p = 1./(1+math.exp(-dp))
            return pclip(p), random.random()
        elif self.method == 'constant':
            return 1, random.random()  # mean p, 0.859
        else:
            raise Exception

    def train_update(self, inst):
        if self.method == 'hlr':
            base = 2.
            p, h = self.predict(inst, base)
            dlp_dw = 2.*(p-inst.p)*(LN2**2)*p*(inst.t/h)
            dlh_dw = 2.*(h-inst.h)*LN2*h
            for (k, x_k) in inst.fv:
                rate = (1./(1+inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                # rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == 'leitner' or self.method == 'pimsleur' or self.method == "my_pimsleur" or self.method == "constant":
            pass
        elif self.method == 'lr':
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                # rate = (1./(1+inst.p)) * self.lrate   / math.sqrt(1 + self.fcounts[k])
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1

    def train(self, trainset):
        if self.method == 'leitner' or self.method == 'pimsleur' or self.method == "my_pimsleur" or self.method == "constant":
            return
        num_iterations = 1  # quick fix  
        for _ in range(num_iterations):
            random.shuffle(trainset)
            for inst in trainset:
                self.train_update(inst)

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p)**2
        slh = (inst.h - h)**2
        return slp, slh, p, h

    def eval(self, testset, prefix=''):
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for inst in testset:
            slp, slh, p, h = self.losses(inst)
            results['p'].append(inst.p)     # ground truth
            results['h'].append(inst.h)
            results['pp'].append(p)         # predictions
            results['hh'].append(h)
            results['slp'].append(slp)      # loss function values
            results['slh'].append(slh)
        mae_p = mae(results['p'], results['pp'])
        mae_h = mae(results['h'], results['hh'])
        cor_p = spearmanr(results['p'], results['pp'])
        cor_h = spearmanr(results['h'], results['hh'])
        total_slp = sum(results['slp'])
        total_slh = sum(results['slh'])
        total_l2 = sum([x**2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt*total_slh + self.l2wt*total_l2
        if prefix:
            sys.stderr.write('%s\t' % prefix)
        sys.stderr.write('%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor(h)=%.3f\n' % \
            (total_loss, total_slp, self.hlwt*total_slh, self.l2wt*total_l2, \
            mae_p, cor_p, mae_h, cor_h))

    def dump_weights(self, fname):
        with open(fname, 'wt') as f:
            for (k, v) in self.weights.items():
                f.write('%s\t%.4f\n' % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, 'wt') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write('%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\n' % (inst.p, pp, inst.h, hh, inst.lang, inst.uid, inst.ts))

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, 'wt') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\tlexeme_tag\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                for i in range(inst.right):
                    f.write('1.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))
                for i in range(inst.wrong):
                    f.write('0.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def mean(lst):
    # the average of a list
    return float(sum(lst))/len(lst)


def spearmanr(l1, l2):
    # spearman rank correlation
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.
    d1 = 0.
    d2 = 0.
    for i in range(len(l1)):
        num += (l1[i]-m1)*(l2[i]-m2)
        d1 += (l1[i]-m1)**2
        d2 += (l2[i]-m2)**2
    try:
        return num/math.sqrt(d1*d2)
    except ZeroDivisionError:
        return 0


def read_data(input_file, method, omit_bias, omit_lexemes, omit_right_wrong, max_lines):
    # read learning trace data in specified format, see README for details
    sys.stderr.write('reading data...')
    instances = list()
    if input_file.endswith('gz'):
        f = gzip.open(input_file, 'rt')
    else:
        f = open(input_file, 'rb')
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row['p_recall']))
        t = float(row['delta'])/(60*60*24)  # convert time delta to days
        h = hclip(-t/(math.log(p, 2)))
        lang = '%s->%s' % (row['ui_language'], row['learning_language'])
        lexeme_id = row['lexeme_id']
        lexeme_string = row['lexeme_string']
        timestamp = int(row['timestamp'])
        user_id = row['user_id']
        seen = int(row['history_seen'])
        right = int(row['history_correct'])
        wrong = seen - right
        right_this = int(row['session_correct'])
        wrong_this = int(row['session_seen']) - right_this
        # feature vector is a list of (feature, value) tuples
        fv = []
        # core features based on method
        if method == 'leitner':
            fv.append((sys.intern('diff'), right-wrong))
        elif method == 'pimsleur' or method == "my_pimsleur":
            fv.append((sys.intern('total'), right+wrong))
        elif method in ["hlr", "lr"]:
            if not omit_right_wrong or omit_bias:
                # fv.append((intern('right'), right))
                # fv.append((intern('wrong'), wrong))
                fv.append((sys.intern('right'), math.sqrt(1+right)))
                fv.append((sys.intern('wrong'), math.sqrt(1+wrong)))
        else:  # constant
            pass
        # optional flag features
        if method == 'lr':
            fv.append((sys.intern('time'), t))
        if not omit_bias:
            fv.append((sys.intern('bias'), 1.))
        if not omit_lexemes:
            fv.append((sys.intern('%s:%s' % (row['learning_language'], lexeme_string)), 1.))
        instances.append(Instance(p, t, fv, h, (right+2.)/(seen+4.), lang, right_this, wrong_this, timestamp, user_id, lexeme_string))
        if i % 1000000 == 0:
            sys.stderr.write('%d...' % i)
    sys.stderr.write('done!\n')
    splitpoint = int(0.9 * len(instances))
    return instances[:splitpoint], instances[splitpoint:]


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-b', action="store_true", default=False, help='omit bias feature')
argparser.add_argument('-l', action="store_true", default=False, help='omit lexeme features')
argparser.add_argument('-t', action="store_true", default=False, help='omit half-life term')
argparser.add_argument('-rw', action="store_true", default=False, help='omit right and wrong terms (for hlr only)')
argparser.add_argument('-m', action="store", dest="method", default='hlr', help="hlr, lr, leitner, pimsleur, my_pimsleur, constant")
argparser.add_argument('-x', action="store", dest="max_lines", type=int, default=None, help="maximum number of lines to read (for dev)")
argparser.add_argument('-v', action="store_true", default="False", help='visualize')
argparser.add_argument('input_file', action="store", help='log file for training')


if __name__ == "__main__":

    args = argparser.parse_args()

    # model diagnostics
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.b:
        sys.stderr.write('--> omit_bias\n')
    if args.l:
        sys.stderr.write('--> omit_lexemes\n')
    if args.t:
        sys.stderr.write('--> omit_h_term\n')
    if args.v:
        sys.stderr.write('--> visualize\n')

    if not args.v:
    # read data set
        trainset, testset = read_data(args.input_file, args.method, args.b, args.l, args.rw, args.max_lines)
        sys.stderr.write('|train| = %d\n' % len(trainset))
        sys.stderr.write('|test|  = %d\n' % len(testset))
        
        # train model & print preliminary evaluation info
        model = SpacedRepetitionModel(method=args.method, omit_h_term=args.t)
        model.train(trainset)
        model.eval(testset, 'test')
        
        # write out model weights and predictions
        filebits = [args.method] + \
            [k for k, v in sorted(vars(args).items()) if v is True] + \
            [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
        if args.max_lines is not None:
            filebits.append(str(args.max_lines))
        filebase = '.'.join(filebits)
        if not os.path.exists('results/'):
            os.makedirs('results/')
        model.dump_weights('results/'+filebase+'.weights')
        model.dump_predictions('results/'+filebase+'.preds', testset)
        # model.dump_detailed_predictions('results/'+filebase+'.detailed', testset)
    else:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        train, test = read_data(args.input_file, "hlr", False, False, False, args.max_lines)
        train_pd, test_pd = pd.DataFrame(train), pd.DataFrame(test)
        train_pd = pd.concat([train_pd, test_pd])
        
        plt.style.use("seaborn")
        plt.figure(figsize=(12, 5))
        # plt.title("Lag time distribution", fontdict={"fontsize": 13, "fontweight": "bold"})
        # plt.title(style)
        plt.hist(train_pd["t"].where(lambda x: x <= 1).dropna(), bins=100)
        plt.xlabel("lag time in days", fontsize=14)
        plt.xlim(0, 1)
        plt.xticks([i / 10 for i in range(0, 10 + 1)])
        plt.ylabel("number of training sessions", fontsize=14)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.savefig("z_histogram_lag_time_dist_1.png", dpi=1000, bbox_inches="tight")
        plt.show()
        
        plt.style.use("seaborn")
        plt.figure(figsize=(12, 5))
        # plt.title("Lag time distribution", fontdict={"fontsize": 13, "fontweight": "bold"})
        plt.hist(train_pd["t"].where(lambda x: x <= 100).dropna(), bins=100)
        plt.xlabel("lag time in days", fontsize=14)
        plt.xlim(0, 100)
        plt.xticks([i * 10 for i in range(0, 10 + 1)])
        plt.ylabel("number of training sessions", fontsize=14)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.savefig("lag_time_dist_100.png", dpi=1000, bbox_inches="tight")
        plt.show()
        
        plt.style.use("seaborn")
        plt.figure(figsize=(12, 5))
        plt.hist(train_pd["p"], bins=100)
        plt.xlabel("observed recall probability " + r"$\hat{p}$", fontsize=14)
        plt.xlim(0, 1)
        plt.xticks([i / 10 for i in range(0, 11)])
        plt.ylabel("number of training sessions", fontsize=14)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
        # plt.savefig("recall_rate_dist.png", dpi=1000, bbox_inches="tight")
        plt.show()
        
        for d in [1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 30]:
            # print(f"days = {d}, number of sessions = {train_pd['t'].where(lambda x: x <= d).dropna().size}")
            pass

