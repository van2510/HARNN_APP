from django.shortcuts import render
from gensim.models import KeyedVectors
from .forms import MyForm
import os
import sys
import time
import logging
import underthesea

f = open('/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/data/stopwords.txt', 'r')
stopwords= f.read()



sys.path.append('/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models')
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import ver2_params as parser
from utils import param_parser as parser3
from utils import ver3_params as parser2



args = parser.parameter_parser()
args3 = parser3.parameter_parser()
args2 = parser2.parameter_parser()

logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime().replace(":", "/")))


def create_input_data(data: dict):
    return zip(data['pad_seqs'], data['onehot_labels'])

def tokenize_content(content):
  tokens = underthesea.word_tokenize(content)
  filtered_tokens = ['~~' + token + '~~' for token in tokens if token not in stopwords]
  return filtered_tokens

def test_harnn_word2vec(filtered_tokens,MODEL,BEST_CPT_DIR):
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_3_level(args, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    true_labels = []
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y_first = graph.get_operation_by_name("input_y_first").outputs[0]
            input_y_second = graph.get_operation_by_name("input_y_second").outputs[0]
            input_y_third = graph.get_operation_by_name("input_y_third").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            third_scores = graph.get_operation_by_name("third-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|third-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            #predicted_onehot_scores = [[], [], [], []]
            #predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores, batch_second_scores, batch_third_scores = \
                    sess.run([scores, first_scores, second_scores, third_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores, batch_second_scores,
                                batch_third_scores]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
    logger.info("All Done.")
    return predicted_labels

def test_harnn_word2vec_2level(filtered_tokens,MODEL,BEST_CPT_DIR):
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_2_levels(args2, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args2.allow_soft_placement,
            log_device_placement=args2.log_device_placement)
        session_conf.gpu_options.allow_growth = args2.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args2.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            #predicted_onehot_scores = [[], [], [], []]
            #predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores, batch_second_scores = \
                    sess.run([scores, first_scores, second_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores, batch_second_scores,
                                ]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args2.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
    logger.info("All Done.")
    return predicted_labels

def test_harnn_fasttext(filtered_tokens,MODEL,BEST_CPT_DIR):

    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_fasttext_matrix(args.fasttext_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_3_level(args, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    true_labels = []
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y_first = graph.get_operation_by_name("input_y_first").outputs[0]
            input_y_second = graph.get_operation_by_name("input_y_second").outputs[0]
            input_y_third = graph.get_operation_by_name("input_y_third").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            third_scores = graph.get_operation_by_name("third-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|third-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            predicted_onehot_scores = [[], [], [], []]
            predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores, batch_second_scores, batch_third_scores = \
                    sess.run([scores, first_scores, second_scores, third_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores, batch_second_scores,
                                batch_third_scores]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
    logger.info("All Done.")
    return predicted_labels
def test_harnn_fasttext_2level(filtered_tokens,MODEL,BEST_CPT_DIR):
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_fasttext_matrix(args.fasttext_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_2_levels(args2, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args2.allow_soft_placement,
            log_device_placement=args2.log_device_placement)
        session_conf.gpu_options.allow_growth = args2.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args2.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            #predicted_onehot_scores = [[], [], [], []]
            #predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores, batch_second_scores = \
                    sess.run([scores, first_scores, second_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores, batch_second_scores,
                                ]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args2.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
    logger.info("All Done.")
    return predicted_labels


def test_harnn_word2vec2(filtered_tokens,MODEL,BEST_CPT_DIR):
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args3, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args3.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_domain(args3, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args3.allow_soft_placement,
            log_device_placement=args3.log_device_placement)
        session_conf.gpu_options.allow_growth = args3.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args3.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            #predicted_onehot_scores = [[], [], [], []]
            #predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args3.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores = \
                    sess.run([scores, first_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args3.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
                print(predicted_labels)
                final_result=[]
                for label in predicted_labels[0]:
                    
                    if label == 0:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684250977','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684250977/bestcheckpoints'))
                        print('0 done')
                    elif label == 1:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684435405','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684435405/bestcheckpoints'))                    
                        print('1 done')

                    elif label == 2:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684330341','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684330341/bestcheckpoints'))
                        print('2 done')

                    elif label == 3:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684435921','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684435921/bestcheckpoints'))
                        print('3 done')
                    elif label == 4:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684338283','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684338283/bestcheckpoints'))
                        print('4 done')

                    elif label == 5:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684344108','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684344108/bestcheckpoints'))
                        print('5 done')

                    elif label == 6:
                        final_result.append(test_harnn_word2vec_2level(filtered_tokens,'1684437296','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684437296/bestcheckpoints'))
                        print('6 done')

                    elif label == 7:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684397290','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684397290/bestcheckpoints'))
                        print('7 done')

                    elif label == 8:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684401826','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684401826/bestcheckpoints'))
                        print('8 done')
                  
                    elif label == 9:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684437642','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684437642/bestcheckpoints'))
                        print('9 done')

                    elif label == 10:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684403602','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684403602/bestcheckpoints'))
                        print('10 done')

                    elif label == 11:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684478909','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684478909/bestcheckpoints'))
                        print('11 done')

                    elif label == 12:
                        final_result.append(test_harnn_word2vec(filtered_tokens,'1684475977','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684475977/bestcheckpoints'))
                        print('12 done')

    return final_result

def test_harnn_fasttext2(filtered_tokens,MODEL,BEST_CPT_DIR):
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args3, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_fasttext_matrix(args.fasttext_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels_domain(args3, filtered_tokens, word2idx)

    # Load harnn model
    logger.info("Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    logger.info(checkpoint_file)
    # Collect the predictions here
    predicted_labels = []
    predicted_scores = []

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args3.allow_soft_placement,
            log_device_placement=args3.log_device_placement)
        session_conf.gpu_options.allow_growth = args3.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.io.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args3.batch_size, 1, shuffle=False)


            # Collect for calculating metrics
            #predicted_onehot_scores = [[], [], [], []]
            #predicted_onehot_labels = [[], [], [], []]

            for batch_test in batches:
                x, y_onehot = zip(*batch_test)

                y_batch_test_list = [y_onehot]

                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args3.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores = \
                    sess.run([scores, first_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args3.threshold)

                # Add results to collection
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)
                print(predicted_labels)
                final_result=[]
                for label in predicted_labels[0]:
                    
                    if label == 0:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684251926','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684251926/bestcheckpoints'))
                        print('0 done')
                    elif label == 1:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684434995','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684434995/bestcheckpoints'))                    
                        print('1 done')

                    elif label == 2:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684332169','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684332169/bestcheckpoints'))
                        print('2 done')

                    elif label == 3:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684436374','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684436374/bestcheckpoints'))
                        print('3 done')
                    elif label == 4:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684341587','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684341587/bestcheckpoints'))
                        print('4 done')

                    elif label == 5:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684342841','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684342841/bestcheckpoints'))
                        print('5 done')

                    elif label == 6:
                        final_result.append(test_harnn_fasttext_2level(filtered_tokens,'1684436932','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684436932/bestcheckpoints'))
                        print('6 done')

                    elif label == 7:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684398762','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684398762/bestcheckpoints'))
                        print('7 done')

                    elif label == 8:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684400142','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684400142/bestcheckpoints'))
                        print('8 done')
                  
                    elif label == 9:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684438019','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684438019/bestcheckpoints'))
                        print('9 done')

                    elif label == 10:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684405062','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684405062/bestcheckpoints'))
                        print('10 done')

                    elif label == 11:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684406500','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684406500/bestcheckpoints'))
                        print('11 done')

                    elif label == 12:
                        final_result.append(test_harnn_fasttext(filtered_tokens,'1684475309','/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684475309/bestcheckpoints'))
                        print('12 done')

    return final_result

hierarchical_labels = {
    0: {'name': 'Thời sự', 'children': {
        13: {'name': 'Chính trị'},
        14: {'name': 'Dân sinh'},
        15: {'name': 'Lao động Việc làm'},
        16: {'name': 'Giao thông'},
        17: {'name': 'Mekong', 'children': {
            79: {'name': 'Đầu tư'},
            80: {'name': 'Nông nghiệp'},
            81: {'name': 'Khám phá'},
        }},
        18: {'name': 'Quỹ Hy vọng'},
    }},
    1: {'name': 'Thế giới', 'children': {
        19: {'name': 'Tư liệu'},
        20: {'name': 'Phân tích'},
        21: {'name': 'Người Việt 5 châu'},
        22: {'name': 'Cuộc sống đó đây'},
        23: {'name': 'Quân sự'},
    }},
    2: {'name': 'Kinh doanh', 'children': {
        24: {'name': 'Quốc tế'},
        25: {'name': 'Doanh nghiệp', 'children': {
            87: {'name': 'Dự án'},
        }},
        26: {'name': 'Chứng khoáng'},
        27: {'name': 'Bất động sản'},
        28: {'name': 'Ebank', 'children': {
            88: {'name': 'Ngân hàng'},
            89: {'name': 'Thanh toán điện tử'},
            90: {'name': 'Tuyển dụng'},
            91: {'name': 'Cộng đồng'},
            92: {'name': 'Tư vấn'},
        }},
        29: {'name': 'Vĩ mô'},
        30: {'name': 'Tiền của tôi', 'children': {
            93: {'name': 'Kinh nghiệm'},
        }},
        31: {'name': 'Bảo hiểm', 'children': {
            94: {'name': 'Tin tức'},
            95: {'name': 'Câu chuyện bảo hiểm'},
            96: {'name': 'Tư vấn'},
        }},
        32: {'name': 'Hàng hoá', 'children': {
            97: {'name': 'Sản phẩm mới'},
            98: {'name': 'Khuyến mãi'},
        }},
    }},
    3: {'name': 'Khoa học', 'children': {
        33: {'name': 'Khoa học trong nước'},
        34: {'name': 'Tin tức'},
        35: {'name': 'Phát minh'},
        36: {'name': 'Ứng dụng'},
        37: {'name': 'Thế giới tự nhiên'},
    }},
    4: {'name': 'Giải trí', 'children': {
        39: {'name': 'Giới sao', 'children': {
            99: {'name': 'Trong nước'},
            100: {'name': 'Quốc tế'},
        }},
        40: {'name': 'Phim', 'children': {
            101: {'name': 'Chuyện màn ảnh'},
        }},
        41: {'name': 'Nhạc', 'children': {
            102: {'name': 'Làng nhạc'},
        }},
        42: {'name': 'Thời trang', 'children': {
            103: {'name': 'Làng mốt'},
            104: {'name': 'Bộ sưu tập'},
            105: {'name': 'Sao đẹp Sao xấu'},
        }},
        43: {'name': 'Làm đẹp'},
        44: {'name': 'Sách', 'children': {
            106: {'name': 'Làng văn'},
            107: {'name': 'Tác giả'},
        }},
        45: {'name': 'Sân khấu Mỹ thuật'},
    }},
    5: {'name': 'Thể thao', 'children': {
        46: {'name': 'Bóng đá', 'children': {
            108: {'name': 'Trong nước'},
            109: {'name': 'Ngoại hạng Anh'},
            110: {'name': 'Champions League'},
            111: {'name': 'La Liga'},
            112: {'name': 'Serie A'},
            113: {'name': 'Các giải khác'},
        }},
        47: {'name': 'Bundesliga', 'children': {
            114: {'name': 'Tin tức'},
            115: {'name': 'Bình luận'},
            116: {'name': 'Hậu trường'},
            117: {'name': 'Tường thuật'},
        }},
        48: {'name': 'Tennis'},
        49: {'name': 'Marathon', 'children': {
            118: {'name': 'Tin tức'},
            119: {'name': 'Kinh nghiệm'},
        }},
        50: {'name': 'Các môn khác', 'children': {
            120: {'name': 'Đua xe'},
            121: {'name': 'Golf'},
            122: {'name': 'Cờ vua'},
            123: {'name': 'Điền kinh'},
        }},
        51: {'name': 'Hậu trường'},
        52: {'name': 'Tường thuật'},
    }},
    6: {'name': 'Pháp luật', 'children': {
        53: {'name': 'Hồ sơ phá án'},
    }},
    7: {'name': 'Giáo dục', 'children': {
        54: {'name': 'Tin tức'},
        55: {'name': 'Tuyển sinh', 'children': {
            124: {'name': 'Phổ thông'},
            125: {'name': 'Đại học'},
        }},
        56: {'name': 'Chân dung'},
        57: {'name': 'Du học'},
        58: {'name': 'Học tiếng Anh'},
        59: {'name': 'Giáo dục 4.0'},
    }},
    8: {'name': 'Sức khoẻ', 'children': {
        60: {'name': 'Tin tức'},
        61: {'name': 'Dinh dưỡng', 'children': {
            126: {'name': 'Thực đơn'},
        }},
        62: {'name': 'Khỏe đẹp', 'children': {
            127: {'name': 'Giảm cân'},
            128: {'name': 'Thẩm mỹ'},
        }},
        63: {'name': 'Đàn ông', 'children': {
            129: {'name': 'Tập luyện'},
            130: {'name': 'Nam khoa'},
            131: {'name': 'Ăn uống'},
        }},
        64: {'name': 'Các bệnh'},
    }},
    9: {'name': 'Đời sống', 'children': {
         65: {'name': 'Nhịp sống'},
         66: {'name': 'Tổ ấm'},
         67: {'name': 'Bài học sống'},
         68: {'name': 'Nhà'},
         69: {'name': 'Tiêu dùng'},
    }},
    10: {'name': 'Du lịch', 'children': {
        70: {'name': 'Điểm đến', 'children': {
            132: {'name': 'Việt Nam'},
            133: {'name': 'Quốc tế'},
        }},
        71: {'name': 'Ẩm thực', 'children': {
            134: {'name': 'Việt Nam'},
            135: {'name': 'Quốc tế'},
        }},
        72: {'name': 'Dấu chân'},
        73: {'name': 'Tư vấn', 'children': {
            136: {'name': 'Đi đâu'},
            137: {'name': 'Ăn gì'},
            138: {'name': 'Kinh nghiệm'},
        }},
    }},
    11: {'name': 'Số hoá', 'children': {
        74: {'name': 'Công nghệ', 'children': {
            139: {'name': 'Đời sống số'},
            140: {'name': 'AI'},
            141: {'name': 'Nhân vật'},
            142: {'name': 'Xu hướng'},
            143: {'name': 'Bảo mật'},
        }},
        75: {'name': 'Sản phẩm', 'children': {
            144: {'name': 'Thiết bị'},
            145: {'name': 'Ứng dụng'},
            146: {'name': 'Đánh giá'},
            147: {'name': 'Thị trường'},
            148: {'name': 'Kinh nghiệm'},
        }},
        76: {'name': 'Blockchain'},
    }},
    12: {'name': 'Xe', 'children': {
        77: {'name': 'Thị trường', 'children': {
            149: {'name': 'Trong nước'},
            150: {'name': 'Thế giới'},
            151: {'name': 'Xe điện'},
        }},
        78: {'name': 'Cầm lái', 'children': {
            152: {'name': 'Đánh giá xe'},
            153: {'name': 'Kỹ năng lái'},
            154: {'name': 'Lật giao thông'},
            155: {'name': 'Chăm sóc xe'},
            156: {'name': 'Độ xe'},
        }},
    }},
}
def find_label_name(label_id, labels_dict):
    label_info = labels_dict.get(label_id)
    if label_info is not None:
        return label_info['name']
    for parent_info in labels_dict.values():
        children = parent_info.get('children')
        if children:
            label_name = find_label_name(label_id, children)
            if label_name:
                return f"{parent_info['name']}/{label_name}"
    return None

def get_label_hierarchy_with_parent(input_labels, labels_dict):
    result = []

    for label_id in input_labels:
        label_name = find_label_name(label_id, labels_dict)
        if label_name:
            result.append(label_name)

    return result


def get_label(input):
    hierarchical_labels_flat = {
        0: 'Thời sự', 13: 'Chính trị', 14: 'Dân sinh', 15: 'Lao động Việc làm', 16: 'Giao thông', 17: 'Mekong', 79: 'Đầu tư', 80: 'Nông nghiệp', 81: 'Khám phá', 18: 'Quỹ Hy vọng',
        1: 'Thế giới', 19: 'Tư liệu', 20: 'Phân tích', 21: 'Người Việt 5 châu', 22: 'Cuộc sống đó đây', 23: 'Quân sự',
        2: 'Kinh doanh', 24: 'Quốc tế', 25: 'Doanh nghiệp', 87: 'Dự án', 26: 'Chứng khoáng', 27: 'Bất động sản', 28: 'Ebank', 88: 'Ngân hàng', 89: 'Thanh toán điện tử', 90: 'Tuyển dụng', 91: 'Cộng đồng', 92: 'Tư vấn', 29: 'Vĩ mô', 93: 'Tiền của tôi',
        30: 'Kinh nghiệm', 31: 'Bảo hiểm', 94: 'Tin tức', 95: 'Câu chuyện bảo hiểm', 96: 'Tư vấn', 32: 'Hàng hoá', 97: 'Sản phẩm mới', 98: 'Khuyến mãi',
        3: 'Khoa học', 33: 'Khoa học trong nước', 34: 'Tin tức', 35: 'Phát minh', 36: 'Ứng dụng', 37: 'Thế giới tự nhiên',
        4: 'Giải trí', 39: 'Giới sao', 99: 'Trong nước', 100: 'Quốc tế', 40: 'Phim', 101: 'Chuyện màn ảnh', 41: 'Nhạc', 102: 'Làng nhạc', 42: 'Thời trang', 103: 'Làng mốt', 104: 'Bộ sưu tập', 105: 'Sao đẹp Sao xấu', 43: 'Làm đẹp', 44: 'Sách', 106: 'Làng văn', 107: 'Tác giả', 45: 'Sân khấu Mỹ thuật',
        5: 'Thể thao', 46: 'Bóng đá', 108: 'Trong nước', 109: 'Ngoại hạng Anh', 110: 'Champions League', 111: 'La Liga', 112: 'Serie A', 113: 'Các giải khác', 47: 'Bundesliga', 114: 'Tin tức', 115: 'Bình luận', 116: 'Hậu trường', 117: 'Tường thuật', 48: 'Tennis', 49: 'Marathon', 118: 'Tin tức', 119: 'Kinh nghiệm', 50: 'Các môn khác', 120: 'Đua xe', 121: 'Golf', 122: 'Cờ vua', 123: 'Điền kinh', 51: 'Hậu trường', 52: 'Tường thuật',
        6: 'Pháp luật', 53: 'Hồ sơ phá án',
        7: 'Giáo dục', 54: 'Tin tức', 55: 'Tuyển sinh', 124: 'Phổ thông', 125: 'Đại học', 56: 'Chân dung', 57: 'Du học', 58: 'Học tiếng Anh', 59: 'Giáo dục 4.0',
        8: 'Sức khoẻ', 60: 'Tin tức', 61: 'Dinh dưỡng', 126: 'Thực đơn', 62: 'Khỏe đẹp', 127: 'Giảm cân', 128: 'Thẩm mỹ', 63: 'Đàn ông', 129: 'Tập luyện', 130: 'Nam khoa', 131: 'Ăn uống', 64: 'Các bệnh',
        9: 'Đời sống', 65: 'Nhịp sống', 66: 'Tổ ấm', 67: 'Bài học sống', 68: 'Nhà', 69: 'Tiêu dùng',
        10: 'Du lịch', 70: 'Điểm đến', 132: 'Việt Nam', 133: 'Quốc tế', 71: 'Ẩm thực', 134: 'Việt Nam', 135: 'Quốc tế', 72: 'Dấu chân', 73: 'Tư vấn', 136: 'Đi đâu', 137: 'Ăn gì', 138: 'Kinh nghiệm',
        11: 'Số hoá', 74: 'Công nghệ', 139: 'Đời sống số', 140: 'AI', 141: 'Nhân vật', 142: 'Xu hướng', 143: 'Bảo mật', 75: 'Sản phẩm', 144: 'Thiết bị', 145: 'Ứng dụng', 146: 'Đánh giá', 147: 'Thị trường', 148: 'Kinh nghiệm', 76: 'Blockchain',
        12: 'Xe', 77: 'Thị trường', 149: 'Trong nước', 150: 'Thế giới', 151: 'Xe điện', 78: 'Cầm lái', 152: 'Đánh giá xe', 153: 'Kỹ năng lái', 154: 'Lật giao thông', 155: 'Chăm sóc xe',
    }
    
    labels = []
    for label_id in input:
        label = hierarchical_labels_flat.get(label_id)
        if label:
            labels.append(label)
        else:
            labels.append(f"Label with ID {label_id} not found")
    
    return labels



#             result += f"<li>{key}"
#             if value:
#                 result += build_nested_list(value)
#             result += "</li>"
#         result += "</ul>"
#         return result

#     return build_nested_list(hierarchy)
def build_hierarchical_structure(input_categories, filter_categories):
    hierarchy = {}

    for category in input_categories:
        parts = category.split('/')
        current_level = hierarchy
        for i, part in enumerate(parts):
            is_last_part = i == len(parts) - 1
            if part not in current_level:
                current_level[part] = {}
            if is_last_part:
                current_level[part]['color'] = 'black'
            current_level = current_level[part]

    def remove_duplicates(current_level, filter_categories):

        if 'color' in current_level and current_level['color'] == 'gray':
            if frozenset(current_level.items()) in filter_categories:
                current_level['color'] = 'black'
            else:
                current_level['color'] = 'gray'
        for key, value in current_level.items():
            if key != 'color':
                remove_duplicates(value, filter_categories)

    for key, value in hierarchy.items():
        remove_duplicates(value,filter_categories)
    def merge_and_remove_duplicates(current_level, parent_key=None):
            for key, value in list(current_level.items()):
                if key.endswith(' '):
                    new_key = key.strip()
                    if new_key not in current_level:
                        current_level[new_key] = {}
                    current_level[new_key].update(value)
                    del current_level[key]

                if key != 'color':
                    merge_and_remove_duplicates(value, key)
                    
    merge_and_remove_duplicates(hierarchy)
    def modify_hierarchy(hierarchy):
        if isinstance(hierarchy, dict):
            for item, info in hierarchy.items():
                if 'color' not in info:
                    info['color'] = 'gray'
                if 'children' in info:
                    modify_hierarchy(info['children'])

    def render_tree(hierarchy, parent_color='black'):
        html = ""
        for key, value in hierarchy.items():
            if isinstance(value, dict):
                color = value.get('color', parent_color)
                html += f"<li style='color: {color};'>{key}</li>\n<ul>{render_tree(value, color)}</ul>\n"
            else:
                if key != 'color':  # Check if the key is not 'color'
                    html += f"<li style='color: {parent_color};'>{key}</li>\n"
        return html

    modify_hierarchy(hierarchy)
    return f"<ul>{render_tree(hierarchy)}</ul>"

def classify_text(request):
    if request.method == 'POST':
        embedding = request.POST.get('embedding')
        approach = request.POST.get('approach')
        title = request.POST.get('title')
        content = request.POST.get('paragraph')

        merged_text = title + ' ' + content
        tokens = underthesea.word_tokenize(merged_text)
        filtered_tokens = [token for token in tokens if token not in stopwords]


        if approach == 'global':
            # For Global Approach, get the top 1 (Word2Vec) or top 2 (FastText) most similar words
            #if mean_vector is not None:
                if embedding == 'word2vec':
                    MODEL = '1685365367'
                    BEST_CPT_DIR = '/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1685365367/bestcheckpoints'
                    temp=test_harnn_word2vec(filtered_tokens,MODEL,BEST_CPT_DIR)[0]
                    print(temp)
                    classification_result =  get_label_hierarchy_with_parent(temp,hierarchical_labels)
                    labels=get_label(temp)
                
                elif embedding == 'fasttext':
                    MODEL = '1685454616'
                    BEST_CPT_DIR = '/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1685454616/bestcheckpoints'
                    temp=test_harnn_fasttext(filtered_tokens,MODEL,BEST_CPT_DIR)[0]
                    classification_result =  get_label_hierarchy_with_parent(temp,hierarchical_labels)
                    labels=get_label(temp)


        elif approach == 'domain-based':
            # For Domain-Based Approach, perform your specific classification logic here
            classification_result = "Domain-Based Approach: Example Result"
            if embedding == 'word2vec':
                    MODEL = '1684741682'
                    BEST_CPT_DIR = '/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684741682/bestcheckpoints'
                    temp=test_harnn_word2vec2(filtered_tokens,MODEL,BEST_CPT_DIR)[0][0]
                    classification_result =  get_label_hierarchy_with_parent(temp,hierarchical_labels)
                    labels=get_label(temp)

            elif embedding == 'fasttext':
                    MODEL = '1684844047'
                    BEST_CPT_DIR = '/Users/van.lambich/Desktop/HARNN_project-1/HARNN_app/models/run/1684844047/bestcheckpoints'
                    temp=test_harnn_fasttext2(filtered_tokens,MODEL,BEST_CPT_DIR)[0][0]
                    classification_result =  get_label_hierarchy_with_parent(temp,hierarchical_labels)
                    labels=get_label(temp)


        else:
            classification_result = "Invalid Approach selected"

        # Pass the classification result to the template
        return render(request, 'form_template.html', {'classification_result': build_hierarchical_structure(classification_result,labels)})

    return render(request, 'form_template.html')
