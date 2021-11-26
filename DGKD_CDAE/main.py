import os
import numpy as np
import tensorflow as tf
from time import time
from util.Parameters import Parameters
from util.Logger import Logger, get_log_dir
from util.ResultTable import ResultTable
from data_loader.Dataset import Dataset
from evaluation.Evaluator import Evaluator

from model.CDAE import CDAE
from model.CDAE_DGKD import CDAE_DGKD


def build_model(sess, dataset, model_name, conf):
    if model_name.lower() == 'cdae':
        model = CDAE(sess, dataset, conf)
    elif model_name.lower() == 'cdae_dgkd':
        model = CDAE_OURS(sess, dataset, conf)
    else:
        raise Exception("Please choose a suitable model.")
    model.build_graph()

    return model


def model_test(model, dataset, evaluator):
    eval_output = model.predict(dataset)
    eval_input = dataset.eval_input
    eval_target = dataset.eval_target
    score = evaluator.compute(eval_input, eval_target, eval_output)
    
    return score


def train_model(sess, model, dataset, evaluator, logger, saver, num_epochs, conf, print_every=1):
    sess.run(tf.compat.v1.global_variables_initializer())
    log_dir = logger.log_dir
    dataset.switch_mode('valid')

    # Train a model.
    early_stop, early_stop_measure, patience = conf.early_stop, conf.early_stop_measure, conf.patience
    endure, best_epoch = 0, -1
    best_score = None
    model_train_start_time = time()

    for epoch in range(1, num_epochs + 1):
        training_start_time = time()
        loss = model.train_model(epoch)
        train_time = time() - training_start_time
        
        # Evaluate after 'test_start_epoch' or if it is the last epoch
        if epoch >= conf.test_start_epoch or epoch == num_epochs:
            # Evaluate the model.
            score, _ = model_test(model, dataset, evaluator)
            if epoch % print_every == 0:
                logger.info("[epoch = %3d, loss = %f, %s, train_time = %.2f]" % (epoch, loss, score, train_time))
            del loss
                
            if best_score is None:
                best_score = score
                best_epoch = epoch
                not_updated = False
            else:
                if score[conf.early_stop_measure] > best_score[conf.early_stop_measure]:
                    best_epoch = epoch
                    best_score = score
                    not_updated = False
                else:
                    not_updated = True
            if not_updated:
                endure += 1
                if conf.early_stop and endure >= conf.patience:
                    logger.info('[%.2fs]Triggered early stopping condition at epoch %d.' % (time() - model_train_start_time,epoch))
                    break
            else:
                if conf.model_name == 'CDAE':
                    saver.save(sess, './ckpt/best_model')
                endure = 0
            
    logger.info('\n')
    return best_epoch, best_score, train_time

if __name__ == "__main__":
    # Read config files.
    conf = Parameters(json_path='main_config.json', name='Exp. Conf')
    model_conf = Parameters(json_path=os.path.join('model_config', conf.model_name + '.json'), name='Model Conf')

    np.random.seed(model_conf.seed)
    tf.random.set_seed(model_conf.seed)
    
    # TensorFlow Session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # TensorFlow components
    dataset = Dataset(conf.data_dir, conf.dataset, conf.separator, conf.implicit, conf.split_type)
    model = build_model(sess, dataset, conf.model_name, model_conf)
    evaluator = Evaluator(conf.split_type, conf.topK, conf.num_threads)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())

    # Build a logger.
    log_dir = get_log_dir(os.path.join('saves', conf.model_name))
    logger = Logger(log_dir, 'experiments.log')

    # Log dataset config.
    logger.info(conf)
    logger.info(model_conf)
    logger.info(dataset)

    # Train model
    best_epoch, best_valid_score, model_train_time = train_model(sess, model, dataset, evaluator, logger, saver, model_conf.num_epochs, conf)
    print("Training time: ", model_train_time)
    test_start_time = time()  
    
    # Restore best parameters
    ckpt = tf.compat.v1.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Evaluate on test data
    #test_start_time = time()  
    dataset.switch_mode('test')
    test_score, test_time = model_test(model, dataset, evaluator)
    score_str = ', '.join(['%s = %.4f' % (k, test_score[k]) for k in test_score])
    line = str(conf.model_name) + "\t" + score_str + "\t" + str(test_time) + "\n"
    print(line, "Inf time: ", time()-test_start_time)
    
    # Make table and log
    header = list(test_score.keys())
    table = ResultTable(header)
    table.add_row('Valid Score', best_valid_score)
    table.add_row('Test Score', test_score)
    logger.info(table.to_string())