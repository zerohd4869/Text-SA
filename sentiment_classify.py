import argparse
import ast
import logging
import math
import os
import sys
import time

import numpy as np
import paddle.fluid as fluid
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import utils
from nets import bilstm_net
from nets import bow_net
from nets import cnn_net
from nets import gru_net
from nets import lstm_net

logger = logging.getLogger("paddle-fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Sentiment Classification.")
    # training data path
    parser.add_argument(
        "--train_data_path",
        type=str,
        default='./kesci_data/train_data.csv',
        required=False,
        help="The path of trainning data. Should be given in train mode!")
    # test data path
    parser.add_argument(
        "--test_data_path",
        type=str,
        default='./kesci_data/validation_data.csv',
        required=False,
        help="The path of test data. Should be given in eval or infer mode!")
    # word_dict path
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default='./kesci_data/train.vocab',
        required=False,
        help="The path of word dictionary.")
    # result data path
    parser.add_argument(
        "--result_path",
        type=str,
        default='./kesci_data/result.csv',
        required=False,
        help="The path of result data.")
    # current mode
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
        required=False,
        choices=['train', 'eval', 'infer'],
        help="train/eval/infer mode")
    # model type
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn_net",
        required=False,
        help="type of model")
    # model save path
    parser.add_argument(
        "--model_path",
        type=str,
        default='./kesci_model/cnn_net_' + str(int(time.time())),
        required=False,
        help="The path to saved the trained models.")
    # Number of passes for the training task.
    parser.add_argument(
        "--num_passes",
        type=int,
        default=50,
        required=False,
        help="Number of passes for the training task.")
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        required=False,
        help="The number of training examples in one forward/backward pass.")
    # lr value for training
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        required=False,
        help="The lr value for training.")
    # Whether to use gpu
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to use gpu to train the model.")
    # parallel train
    parser.add_argument(
        "--is_parallel",
        type=ast.literal_eval,
        default=False,
        help="Whether to train the model in parallel.")
    return parser.parse_args()


def train_net(train_reader,
              test_reader,
              word_dict,
              network,
              use_gpu,
              parallel,
              save_dirname,
              lr=0.002,
              batch_size=128,
              pass_num=30,
              is_local=True):
    """
    train network
    """
    if network == "bilstm_net":
        network = bilstm_net
    elif network == "bow_net":
        network = bow_net
    elif network == "cnn_net":
        network = cnn_net
    elif network == "lstm_net":
        network = lstm_net
    elif network == "gru_net":
        network = gru_net
    else:
        print("unknown network type")
        return
    # 输入层 word seq data
    raw_id = fluid.layers.data(name="raw_id", shape=[1], dtype="int64")
    # lod_level不为0指定输入数据为序列数据
    wids = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    # text = fluid.layers.data(
    #     name="text", shape=[-1,400],dtype="float32", lod_level=1)
    # 标签层 label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 网络结构
    avg_cost, acc, auc_out, pred = network(wids, label, len(word_dict) + 1)

    # test program
    test_program = fluid.default_main_program().clone(for_test=True)

    # 优化器 set optimizer
    optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    optimizer.minimize(avg_cost)

    # 设备、执行器、feeder 定义
    # set place, executor, datafeeder
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[raw_id, wids, label], place=place)

    def train_loop(main_program):
        # 仅运行一次startup program. 不需要优化/编译这个startup program.
        exe.run(fluid.default_startup_program())

        # parallelize it
        # train_exe = fluid.ParallelExecutor(use_cuda=use_gpu, loss_name=avg_cost.name) \
        #     if args.is_parallel else exe

        # start training...
        best_auc = 0.0  # For save checkpoint(model)
        last_auc = 0.0
        stop_gate = 0

        for pass_id in range(pass_num):
            for batch_id, train_data in enumerate(train_reader()):
                # train a batch
                exe.run(main_program, feed=feeder.feed(train_data))
                if (batch_id % 10) == 0:
                    acc_list = []
                    auc_list = []
                    avg_loss_list = []

                    # compute class2_acc and class2_auc: step_1
                    # class2_acc, class3_acc = 0.0, 0.0
                    # total_count, neu_count = 0, 0
                    # y, p = [], []
                    for tid, test_data in enumerate(test_reader()):
                        loss_t, acc_t, auc_out_t, pred_t = exe.run(program=test_program,
                                                                   feed=feeder.feed(test_data),
                                                                   fetch_list=[avg_cost, acc, auc_out, pred])
                        if math.isnan(float(loss_t)):
                            sys.exit("got NaN loss, training failed.")
                        acc_list.append(float(acc_t))
                        auc_list.append(float(auc_out_t))
                        avg_loss_list.append(float(loss_t))
                        # break  # Use 1 segment for speeding up CI

                        # compute class2_acc and class2_auc: step_2
                    #     for i, val in enumerate(test_data):
                    #         _, class2_label = utils.get_predict_label(pred_t[i, 1])
                    #         true_label = val[2]
                    #         if class2_label == true_label:
                    #             class2_acc += 1
                    #         if true_label == 1.0:
                    #             neu_count += 1
                    #         y.append(true_label)
                    #         p.append(pred_t[i, 1])
                    #     total_count += len(test_data)
                    # fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)
                    # class2_auc = auc(fpr, tpr)
                    # class2_acc = class2_acc / total_count

                    acc_value = np.array(acc_list).mean()
                    auc_value = np.array(auc_list).mean()
                    avg_loss_value = np.array(avg_loss_list).mean()
                    print("[train info]: pass_id: %d, batch_id: %d, avg_acc: %.3f, avg_auc: %.3f, avg_cost: %.3f    " %
                          (pass_id, batch_id, float(acc_value), float(auc_value), float(avg_loss_value)))

                    # model checkpoint
                    if abs(auc_value - last_auc) < 0.0008 and auc_value < last_auc:
                        stop_gate += 1
                    last_auc = auc_value
                    if (best_auc < auc_value) & (0.85 < auc_value) & (0.75 < acc_value):
                        stop_gate = 1
                        best_auc = auc_value
                        epoch_model_dir = save_dirname + "/" + "epoch" + str(pass_id) + "-{:.3g}".format(best_auc) + ".model"
                        fluid.io.save_inference_model(epoch_model_dir, ["words"], pred, exe)
                        print("Saved model checkpoint to {}".format(epoch_model_dir))

            if stop_gate > 10:
                print("Early Stopping. Bye~")
                break

    if is_local:
        # fluid.default_main_program 模型参数初始化 initilize parameters
        train_loop(fluid.default_main_program())


def eval_net(test_reader, use_gpu, model_path=None):
    """
    Evaluation function
    """
    if model_path is None:
        print(str(model_path) + "can not be found")
        return
    if os.path.isdir(model_path):
        dir_list = os.listdir(model_path)
        if '.DS_Store' in dir_list:
            dir_list.remove('.DS_Store')
        dir_list.sort(key=lambda x: int(x.split('-')[0][5:]))
        for j in range(0, len(dir_list)):
            path = os.path.join(model_path, dir_list[j])
            if os.path.isdir(path):
                # set place, executor
                place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
                exe = fluid.Executor(place)

                inference_scope = fluid.core.Scope()
                with fluid.scope_guard(inference_scope):
                    # load the saved model
                    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(path, exe)

                    # compute 2class and 3class accuracy
                    class2_acc, _ = 0.0, 0.0
                    total_count, neu_count = 0, 0
                    y, p = [], []
                    for data in test_reader():
                        # infer a batch
                        pred = exe.run(inference_program,
                                       feed=utils.data2tensor(data, place),
                                       fetch_list=fetch_targets,
                                       return_numpy=True)
                        for j, val in enumerate(data):
                            _, class2_label = utils.get_predict_label(pred[0][j, 1])
                            true_label = val[2]
                            if class2_label == true_label:
                                class2_acc += 1
                            if true_label == 1.0:
                                neu_count += 1
                            y.append(true_label)
                            p.append(pred[0][j, 1])
                        total_count += len(data)
                    fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)
                    class2_auc = auc(fpr, tpr)
                    class2_acc = class2_acc / total_count
                    print("[test info] model_path: %s, class2_acc: %f, class2_auc: %f " %
                          (path, class2_acc, class2_auc))


def infer_net(test_reader, result_path, use_gpu, model_path=None):
    """
    Inference function
    """
    if model_path is None:
        print(str(model_path) + "can not be found")
        return
    # set place, executor
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    label = []
    ID = []
    with fluid.scope_guard(inference_scope):
        # load the saved model
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        for data in test_reader():
            # infer a batch
            pred = exe.run(inference_program,
                           feed=utils.data2tensor(data, place),
                           fetch_list=fetch_targets,
                           return_numpy=True)
            for i, val in enumerate(data):
                _, class2_label = utils.get_predict_label(pred[0][i, 1])
                pos_prob = pred[0][i, 1]
                neg_prob = 1 - pos_prob
                print("predict label: %s, pos_prob: %f, neg_prob: %f" %
                      (class2_label, pos_prob, neg_prob))
                label.append(pos_prob)
                ID.append(val[0])
    df = {'ID': ID, 'Pred': label}
    pd.DataFrame(df, index=ID).to_csv(result_path, index=None)


def main(args):
    # train mode
    if args.mode == "train":
        # prepare_data to get word_dict, train_reader
        word_dict, train_reader, test_reader = utils.prepare_data(
            args.train_data_path_list, args.word_dict_path, args.batch_size, args.mode)

        train_net(
            train_reader,
            test_reader,
            word_dict,
            args.model_type,
            args.use_gpu,
            args.is_parallel,
            args.model_path,
            args.lr,
            args.batch_size,
            args.num_passes)

    # eval mode
    elif args.mode == "eval":
        # prepare_data to get word_dict, test_reader    
        word_dict, test_reader = utils.prepare_data(
            args.test_data_path, args.word_dict_path, args.batch_size, args.mode)
        eval_net(
            test_reader,
            args.use_gpu,
            args.model_path)

    # infer mode
    elif args.mode == "infer":
        # prepare_data to get word_dict, test_reader
        word_dict, test_reader = utils.prepare_data(
            args.test_data_path, args.word_dict_path, args.batch_size, args.mode)
        infer_net(
            test_reader,
            args.result_path,
            args.use_gpu,
            args.model_path)


if __name__ == "__main__":
    args = parse_args()

    args.mode = 'train'  # train, infer, eval
    args.train_data_path = './kesci_data/train_data.csv'
    args.test_data_path = './kesci_data/validation_data.csv' if args.mode != 'infer' else './kesci_data/20190527_test_processed.csv'
    args.train_data_path_list = [args.train_data_path, args.test_data_path]  # train
    args.result_path = './kesci_data/result.csv'
    args.word_dict_path = './kesci_data/train.vocab'

    args.model_type = 'cnn_net'  # cnn_net gru_net bow_net lstm_net bilstm_net
    args.model_path = './kesci_model/' + args.model_type + '_' + str(int(time.time())) if args.mode == 'train' \
        else './kesci_model/cnn_net_1558978171' if args.mode != 'infer' else './kesci_model/cnn_net_1558978171/epoch35-0.865.model'
    args.is_parallel = False
    args.batch_size = 256
    args.num_passes = 50
    args.lr = 0.001

    print("model_path: %s" % args.model_path)
    main(args)
