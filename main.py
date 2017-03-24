import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import time
import datetime
import load_data as load
import matplotlib.pyplot as plt

cwd = os.getcwd()

# Settings
save_loaction = cwd + '\\model\\model.ckpt'
train_log_location = cwd + '\\log\\train\\'
test_log_location = cwd + '\\log\\test\\'
load_model = True
train_model = False
save_model = False
compute_final_accuracy = True
clear_log = False
create_confusion_matrix_plot = True

if clear_log:
    filelist = [f for f in os.listdir(train_log_location)]
    for f in filelist:
        os.remove(train_log_location + f)

    filelist = [f for f in os.listdir(test_log_location)]
    for f in filelist:
        os.remove(test_log_location + f)

# Hyperparameters
model_cell_type = 'gru'
embedding_size = 8
max_seq_length = 128
rnn_size = 256
hidden_layer_size = 256
n_stacked_cells = 2
n_classes = 4
keep_prob = 0.7
clip_value = None
l2 = 0.0
n_epochs = 30
batch_size = 256
use_full_seq_lengths = True
use_molecular_weights = True
use_hydropathy_seqs = True
use_isoelectric_point_seqs = True
use_pk1_seqs = True
use_pk2_seqs = True
use_fully_connected_layer = False
use_extra_fully_connected_layer = False
if model_cell_type == 'gru':
    learning_rate = 0.005
else:
    learning_rate = 0.0003

# Load data
print('Loading data...')
data = load.read_data(max_length=max_seq_length)
print('Data loaded.')

print('Number of training examples: ' + str(data.num_train_examples))
print('Max truncated sequence length: ' + str(max(data.train.seq_lengths)))

# MODEL
x = tf.placeholder(tf.int32, [None, None])  # [batch_size, seq_length, 1]
y = tf.placeholder('float')
sequence_length = tf.placeholder(tf.int32, [None])
full_sequence_length = tf.placeholder(tf.int32, [None])
molecular_weight = tf.placeholder(tf.int32, [None])
hydropathy_seq = tf.placeholder('float', [None, None])
isoelectric_point_seq = tf.placeholder('float', [None, None])
pk1_seq = tf.placeholder('float', [None, None])
pk2_seq = tf.placeholder('float', [None, None])

input_shape = tf.shape(x)
input_size = input_shape[0]
input_seq_length = input_shape[1]
vocab_size = 23

# Amino acid embeddings
with tf.name_scope('embeddings'):
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    embeddings = tf.get_variable("W", [vocab_size, embedding_size], initializer=initializer, trainable=True)
    tf.summary.histogram('embeddings', embeddings)

with tf.variable_scope("embedders") as varscope:
    sequences_embedded = tf.nn.embedding_lookup(embeddings, x)  # [batch_size, seq_length, embedding_size]

rnn_input_list = [sequences_embedded]

if use_hydropathy_seqs:
    rnn_input_list.append(tf.expand_dims(hydropathy_seq, 2))

if use_isoelectric_point_seqs:
    rnn_input_list.append(tf.expand_dims(isoelectric_point_seq, 2))

if use_pk1_seqs:
    rnn_input_list.append(tf.expand_dims(pk1_seq, 2))

if use_pk2_seqs:
    rnn_input_list.append(tf.expand_dims(pk2_seq, 2)/2)

rnn_input = tf.concat(rnn_input_list, 2)

# RNN layer
with tf.name_scope('RNN'):
    if model_cell_type == 'lstm':
        rnn_cell = rnn.LSTMCell(rnn_size)
    elif model_cell_type == 'gru':
        rnn_cell = rnn.GRUCell(rnn_size)
    else:
        print('cell type must be lstm or gru')

    if keep_prob < 1:
        rnn_cell_fw = rnn.DropoutWrapper(rnn_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

    if n_stacked_cells > 1:
        rnn_cell_fw = rnn.MultiRNNCell([rnn_cell]*n_stacked_cells)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, rnn_input, sequence_length,
                                                      dtype=tf.float32,
                                                      swap_memory=True)  # outputs - [batch_size, seq_length, hidden_units]
    concat_outputs = tf.concat(outputs, 2)
    rnn_output = concat_outputs[:, -1, :]  # [batch_size, 2*hidden_units]
    tf.summary.tensor_summary('rnn_output', rnn_output)

linear_layer_input_size = rnn_size*2
feature_list = [rnn_output]

if use_full_seq_lengths:
    feature_list.append(tf.cast(tf.expand_dims(full_sequence_length, 1)/130000, tf.float32))
    linear_layer_input_size += 1

if use_molecular_weights:
    feature_list.append(tf.cast(tf.expand_dims(molecular_weight, 1)/707490, tf.float32))
    linear_layer_input_size += 1

linear_layer_input = tf.concat(feature_list, 1)

if use_fully_connected_layer:
    # Linear layer 1
    with tf.name_scope('linear_layer_1'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.truncated_normal([linear_layer_input_size, hidden_layer_size]))
            tf.summary.histogram('weights', w1)
        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.zeros([hidden_layer_size]))
            tf.summary.histogram('biases', b1)
        with tf.name_scope('pre_activations'):
            h1 = tf.nn.relu(tf.matmul(linear_layer_input, w1) + b1)
            tf.summary.histogram('pre_activations', h1)
    linear_layer_1_2_input_size = hidden_layer_size
else:
    h1 = linear_layer_input
    if use_extra_fully_connected_layer:
        linear_layer_1_2_input_size = linear_layer_input_size

if use_extra_fully_connected_layer:
    # Linear layer 1_2
    with tf.name_scope('linear_layer_1_2'):
        with tf.name_scope('weights'):
            w1_2 = tf.Variable(tf.truncated_normal([linear_layer_1_2_input_size, hidden_layer_size]))
            tf.summary.histogram('weights', w1_2)
        with tf.name_scope('biases'):
            b1_2 = tf.Variable(tf.zeros([hidden_layer_size]))
            tf.summary.histogram('biases', b1_2)
        with tf.name_scope('pre_activations'):
            h1_2 = tf.nn.relu(tf.matmul(h1, w1_2) + b1_2)
            tf.summary.histogram('pre_activations', h1_2)

    linear_layer_2_input_size = hidden_layer_size
else:
    h1_2 = h1
    if use_fully_connected_layer:
        linear_layer_2_input_size = hidden_layer_size
    else:
        linear_layer_2_input_size = linear_layer_input_size

# Linear layer 2
with tf.name_scope('linear_layer_2'):
    with tf.name_scope('weights'):
        w2 = tf.Variable(tf.truncated_normal([linear_layer_2_input_size, n_classes]))
        tf.summary.histogram('weights', w2)
    with tf.name_scope('biases'):
        b2 = tf.Variable(tf.zeros([n_classes]))
        tf.summary.histogram('biases', b2)
    with tf.name_scope('pre_activations'):
        h2 = tf.matmul(h1_2, w2) + b2
        tf.summary.histogram('pre_activations', h2)

    prediction = tf.nn.softmax(h2)
    tf.summary.histogram('prediction', prediction)

# Define cost
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=y))
    tf.summary.scalar('cost', cost)

    if l2 != 0.0:
        cost = cost + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

# Define accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

# Define confusion
confusion = tf.contrib.metrics.confusion_matrix(tf.argmax(prediction, 1), tf.argmax(y, 1))

# Define optimiser
with tf.name_scope('train'):
    minimizer = tf.train.AdamOptimizer(learning_rate)
    if clip_value is not None:
        grads_and_vars = minimizer.compute_gradients(cost)
        grad_clipping = tf.constant(clip_value, name="gradient_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))

        # gradient update
        optimiser = minimizer.apply_gradients(clipped_grads_and_vars)
    else:
        optimiser = minimizer.minimize(cost)

# Create saver
saver = tf.train.Saver()

with tf.Session() as sess:
    #with tf.device('/cpu:0'):

    # Initialise variables
    sess.run(tf.global_variables_initializer())

    # Load Model
    if load_model:
        # Load model
        saver.restore(sess, save_loaction)

    # Train
    if train_model:

        # TensorBoard operations
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_log_location, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_location, sess.graph)

        epochs = range(1, n_epochs + 1)

        best_test_accuracy = 0

        epoch_times = []

        for epoch in epochs:
            epoch_start_time = time.time()
            print('********** Epoch ', epoch, ' of ', n_epochs, '**********')

            total_loss = 0

            batch_train_accuracy_sum = 0

            epoch_ind = 0
            for _ in range(int(data.num_train_examples/batch_size)):
                epoch_x, epoch_y, epoch_seq_lengths, epoch_full_seq_lengths, epoch_molecular_weights, epoch_hydropathy_seqs, epoch_isoelectric_point_seqs, epoch_pk1_seqs, epoch_pk2_seqs = data.train.next_batch(batch_size)

                _, c = sess.run([optimiser, cost], feed_dict={x: epoch_x,
                                                              y: epoch_y,
                                                              sequence_length: epoch_seq_lengths,
                                                              full_sequence_length: epoch_full_seq_lengths,
                                                              molecular_weight: epoch_molecular_weights,
                                                              hydropathy_seq: epoch_hydropathy_seqs,
                                                              isoelectric_point_seq: epoch_isoelectric_point_seqs,
                                                              pk1_seq: epoch_pk1_seqs,
                                                              pk2_seq: epoch_pk2_seqs})

                total_loss += c

                batch_train_accuracy = sess.run(accuracy, feed_dict={x: epoch_x,
                                                                     y: epoch_y,
                                                                     sequence_length: epoch_seq_lengths,
                                                                     full_sequence_length: epoch_full_seq_lengths,
                                                                     molecular_weight: epoch_molecular_weights,
                                                                     hydropathy_seq: epoch_hydropathy_seqs,
                                                                     isoelectric_point_seq: epoch_isoelectric_point_seqs,
                                                                     pk1_seq: epoch_pk1_seqs,
                                                                     pk2_seq: epoch_pk2_seqs})

                batch_train_accuracy_sum += batch_train_accuracy

                epoch_ind += 1
                print('Batch ' + str(epoch_ind) + ' of ' + str(np.floor(data.num_train_examples/batch_size)) + ' in epoch. ' + 'Accuracy: ' + str(batch_train_accuracy))

            #train_accuracy = batch_train_accuracy_sum / int((data.num_train_examples / batch_size))

            train_eval_size = data.num_train_examples//6
            train_inputs, train_labels, train_seq_lengths, train_full_seq_lengths, train_molecular_weights, train_hydropathy_seqs, train_isoelectric_point_seqs, train_pk1_seqs, train_pk2_seqs = data.train.next_batch(train_eval_size)
            train_summary, train_accuracy = sess.run([merged, accuracy],
                                                     feed_dict={x: train_inputs,
                                                                y: train_labels,
                                                                sequence_length: train_seq_lengths,
                                                                full_sequence_length: train_full_seq_lengths,
                                                                molecular_weight: train_molecular_weights,
                                                                hydropathy_seq: train_hydropathy_seqs,
                                                                isoelectric_point_seq: train_isoelectric_point_seqs,
                                                                pk1_seq: train_pk1_seqs,
                                                                pk2_seq: train_pk2_seqs})

            test_eval_size = data.num_test_examples
            test_inputs, test_labels, test_seq_lengths, test_full_seq_lengths, test_molecular_weights, test_hydropathy_seqs, test_isoelectric_point_seqs, test_pk1_seqs, test_pk2_seqs = data.test.next_batch(test_eval_size)
            test_summary, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={x: test_inputs,
                                                              y: test_labels,
                                                              sequence_length: test_seq_lengths,
                                                              full_sequence_length: test_full_seq_lengths,
                                                              molecular_weight: test_molecular_weights,
                                                              hydropathy_seq: test_hydropathy_seqs,
                                                              isoelectric_point_seq: test_isoelectric_point_seqs,
                                                              pk1_seq: test_pk1_seqs,
                                                              pk2_seq: test_pk2_seqs})

            # Write to TensorBoard
            train_writer.add_summary(train_summary, epoch)
            saver.save(sess, train_log_location + 'model.ckpt')

            test_writer.add_summary(test_summary, epoch)
            saver.save(sess, test_log_location + 'model.ckpt')

            print('Epoch loss: ', total_loss)
            print('Epoch train accuracy: ', train_accuracy)
            print('Epoch test accuracy: ', test_accuracy)

            if save_model:
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    saver.save(sess, save_loaction)
                    print('Model saved')

            # Compute epoch timing and eta
            epoch_end_time = time.time()
            epoch_times.append(epoch_end_time-epoch_start_time)
            avg_time_per_epoch = sum(epoch_times)/len(epoch_times)
            est_time_left = (n_epochs - epoch)*avg_time_per_epoch

            print('Epoch time: ', str(datetime.timedelta(seconds=round(epoch_times[-1]))))
            print('Est. time remaining: ', str(datetime.timedelta(seconds=round(est_time_left))))

    if compute_final_accuracy:
        # Load best model
        saver.restore(sess, save_loaction)

        # Compute final train and test loss and accuracy
        total_train_loss = 0
        total_train_accuracy = 0
        for _ in range(int(data.num_train_examples / batch_size)):
            epoch_x, epoch_y, epoch_seq_lengths, epoch_full_seq_lengths, epoch_molecular_weights, epoch_hydropathy_seqs, epoch_isoelectric_point_seqs, epoch_pk1_seqs, epoch_pk2_seqs = data.train.next_batch(
                batch_size)
            batch_train_loss, batch_train_accuracy = sess.run([cost, accuracy],
                                                              feed_dict={x: epoch_x,
                                                                         y: epoch_y,
                                                                         sequence_length: epoch_seq_lengths,
                                                                         full_sequence_length: epoch_full_seq_lengths,
                                                                         molecular_weight: epoch_molecular_weights,
                                                                         hydropathy_seq: epoch_hydropathy_seqs,
                                                                         isoelectric_point_seq: epoch_isoelectric_point_seqs,
                                                                         pk1_seq: epoch_pk1_seqs,
                                                                         pk2_seq: epoch_pk2_seqs})

            total_train_loss += batch_train_loss
            total_train_accuracy += batch_train_accuracy

        train_loss = total_train_loss / int(data.num_train_examples / batch_size)
        train_accuracy = total_train_accuracy / int(data.num_train_examples / batch_size)

        total_test_loss = 0
        total_test_accuracy = 0
        total_confusion = np.zeros([4, 4])
        for _ in range(int(data.num_test_examples / batch_size)):
            epoch_x, epoch_y, epoch_seq_lengths, epoch_full_seq_lengths, epoch_molecular_weights, epoch_hydropathy_seqs, epoch_isoelectric_point_seqs, epoch_pk1_seqs, epoch_pk2_seqs = data.test.next_batch(
                batch_size)
            batch_test_loss, batch_test_accuracy = sess.run([cost, accuracy],
                                                            feed_dict={x: epoch_x,
                                                                       y: epoch_y,
                                                                       sequence_length: epoch_seq_lengths,
                                                                       full_sequence_length: epoch_full_seq_lengths,
                                                                       molecular_weight: epoch_molecular_weights,
                                                                       hydropathy_seq: epoch_hydropathy_seqs,
                                                                       isoelectric_point_seq: epoch_isoelectric_point_seqs,
                                                                       pk1_seq: epoch_pk1_seqs,
                                                                       pk2_seq: epoch_pk2_seqs})

            con = sess.run(confusion, feed_dict={x: epoch_x,
                                                 y: epoch_y,
                                                 sequence_length: epoch_seq_lengths,
                                                 full_sequence_length: epoch_full_seq_lengths,
                                                 molecular_weight: epoch_molecular_weights,
                                                 hydropathy_seq: epoch_hydropathy_seqs,
                                                 isoelectric_point_seq: epoch_isoelectric_point_seqs,
                                                 pk1_seq: epoch_pk1_seqs,
                                                 pk2_seq: epoch_pk2_seqs})

            total_confusion += con

            total_test_loss += batch_test_loss
            total_test_accuracy += batch_test_accuracy

        test_loss = total_test_loss / int(data.num_test_examples / batch_size)
        test_accuracy = total_test_accuracy / int(data.num_test_examples / batch_size)

        print('Train loss: ', train_loss)
        print('Test loss: ', test_loss)
        print('Train accuracy: ', train_accuracy)
        print('Test accuracy: ', test_accuracy)

        if create_confusion_matrix_plot:
            import seaborn as sn
            import pandas as pd

            df_cm = pd.DataFrame(con, index=[i for i in ['Cyto', 'Mito', 'Nucleus', 'Secreted']],
                                 columns=[i for i in ['Cyto', 'Mito', 'Nucleus', 'Secreted']])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, fmt="d", cmap="RdBu_r")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            #plt.show()
            plt.savefig(
                cwd,
                format='eps', dpi=1000)


def predict_labels(seqs, sequence_lengths, full_sequence_lengths, molecular_weights, hydropathy_seqs,
                   isoelectric_point_seqs, pk1_seqs, pk2_seqs):
    with tf.Session() as sess:
        # with tf.device('/cpu:0'):

        # Initialise variables
        sess.run(tf.global_variables_initializer())

        # Load model
        saver.restore(sess, save_loaction)

        # Predict
        predictions = sess.run(prediction, feed_dict={x: seqs,
                                                      sequence_length: sequence_lengths,
                                                      full_sequence_length: full_sequence_lengths,
                                                      molecular_weight: molecular_weights,
                                                      hydropathy_seq: hydropathy_seqs,
                                                      isoelectric_point_seq: isoelectric_point_seqs,
                                                      pk1_seq: pk1_seqs,
                                                      pk2_seq: pk2_seqs})

        return predictions


blind_test_predictions = predict_labels(data.blind_test_inputs,
                                        data.blind_test_seq_lengths,
                                        data.blind_test_full_seq_lengths,
                                        data.blind_test_molecular_weights,
                                        data.blind_test_hydropathy_seqs,
                                        data.blind_test_isoelectric_point_seqs,
                                        data.blind_test_pk1_seqs,
                                        data.blind_test_pk2_seqs)

print('Blind test prediction probabilities: ')
print(blind_test_predictions)

prediction_indices = np.argmax(blind_test_predictions, 1)
print('Blind test prediction indices: ')
print(prediction_indices)

labels = {0: 'Cyto', 1: 'Mito', 2: 'Nucl', 3: 'Secr'}

for i, index in enumerate(prediction_indices):
    print(data.blind_test_identifiers[i] + ' ' + labels[prediction_indices[i]] + ' confidence ' + str(100*blind_test_predictions[i, prediction_indices[i]]) + '%')
