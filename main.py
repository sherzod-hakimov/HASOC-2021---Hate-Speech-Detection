import getopt
import sys
import tensorflow as tf
import os
import json
import numpy as np
import file_utils
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
from bert.tokenization.bert_tokenization import FullTokenizer
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
import bert
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, accuracy_score
import data_loader
import models


def prepare_predictions(ids, predictions, labels):
    prediction_output = []
    binary_predictions = list()

    total_expected = {0: 0, 1: 0}
    true_positives = {0: 0, 1: 0}


    for i in range(0, len(labels)):
        predicted_probs = predictions[i]
        predicted_class = 1 if predicted_probs[1] >= 0.51 else 0
        expected = int(labels[i])

        binary_predictions.append(predicted_class)

        if expected == predicted_class:
            true_positives[expected] +=1
        total_expected[expected] += 1


        l = {"id": str(ids[i]), "prediction": str(predicted_class), "label": str(labels[i]), "probs": predicted_probs.tolist()}
        prediction_output.append(json.dumps(l))

    recall_hate = (true_positives[1] / total_expected[1]) if total_expected[1] > 0 else 0
    recall_not_hate = (true_positives[0] / total_expected[0]) if total_expected[0] > 0 else 0

    binary_predictions = np.array(binary_predictions)
    average_precision = average_precision_score(binary_predictions, labels)
    f1 = f1_score(binary_predictions, labels, average='binary')
    f1_weighted = f1_score(binary_predictions, labels, average='weighted')
    macro_f1 = f1_score(binary_predictions, labels, average='macro')
    recall = recall_score(binary_predictions, labels, average='binary')
    precision = precision_score(binary_predictions, labels, average='binary')
    accuracy = accuracy_score(binary_predictions, labels)

    score_output = {"accuracy": accuracy, "average_precision":average_precision, "f1":f1, "weighted_f1":f1_weighted,  "macro_f1":macro_f1,  "recall":recall, "precision":precision,
                    "HatefulOffensive": {"recall": recall_hate, "support": total_expected[1]},
                    "NOT": {"recall": recall_not_hate, "support": total_expected[0]}
                    }

    return prediction_output, score_output

def train(config):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)

    # hate word list
    hate_words = file_utils.read_file_to_list(config['base_dir'] +'resources/hate_words.txt')

    # BERT related configurations
    print('Using BERT: {}'.format(config['bert_model_dir']))
    bert_ckpt_dir =  config['base_dir'] + config['bert_model_dir'] + "/"
    bert_check_point_file = bert_ckpt_dir + "bert_model.ckpt"
    bert_config_file = bert_ckpt_dir + "bert_config.json"
    bert_tokenizer = bert.tokenization.bert_tokenization.FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

    X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids, X_test, y_test, y_test_ids = data_loader.load_dataset(config, bert_tokenizer, hate_words)

    print("Training input file shapes")
    for k in X_train:
        print('\t' + k + " shape: " + str(X_train[k].shape))

    print("Validation input file shapes")
    for k in X_valid:
        print('\t' + k + " shape: " + str(X_valid[k].shape))

    print("Test data size", len(y_test_ids))

    # folders to save the trained models and results

    results_dir_path = config['base_dir']  +'results'
    now = datetime.now()
    model_dir_path = config['base_dir'] +'results/'+now.strftime("%d-%m-%Y %H:%M:%S").replace(" ", "_")

    file_utils.create_folder(results_dir_path)
    file_utils.create_folder(model_dir_path)


    model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(
        model_dir_path + '/best_model-epoch-{epoch:03d}-acc-{acc:03f}-val_acc-{val_acc:03f}.h5',
        save_best_only=True,
        monitor=config['monitor'])

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=config["epoch_patience"],
                                                               restore_best_weights=True,
                                                               monitor=config['monitor'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir_path + "/logs")

    callbacks = [early_stopping_callback]

    print('Using GPUs: ' + str(tf.test.is_gpu_available()))

    # create the model
    model = models.get_model(config, bert_config_file, bert_check_point_file, adapter_size=None)


    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                        batch_size=config['batch_size'],
                        shuffle=True,
                        epochs=config['epochs'],
                        callbacks=callbacks)

    predictions = model.predict(X_test, batch_size=config['batch_size'])

    test_predictions, test_score_output = prepare_predictions(y_test_ids, predictions, y_test['output_label'])

    print('Test macro-f1: ', test_score_output['macro_f1'])

    # save the model
    # model.save(model_dir_path + "/model.h5")


    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(test_score_output),
                                   model_dir_path + '/test_prediction_score.json')
    file_utils.save_list_to_file(test_predictions, model_dir_path + '/test_predictions.jsonl')

    # save the training config
    file_utils.save_string_to_file(json.dumps(config),
                                   model_dir_path + '/training_config.json')


    N = len(history.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, N + 1), history.history['loss'], label='loss')
    plt.plot(np.arange(1, N + 1), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(1, N + 1), history.history['acc'], label='acc')
    plt.plot(np.arange(1, N + 1), history.history['val_acc'], label='val_acc')
    plt.title("Validation, Test Loss and Accuracy on HASOC "+config["dataset_year"]+" Dataset, " + config['optimizer'])
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(model_dir_path + "/history.png")
    plt.close()



if __name__ == "__main__":
    argv = (sys.argv[1:])
    config_path = 'config.json'
    try:
        opts, args = getopt.getopt(argv, "hc:o:")
    except getopt.GetoptError:
        print('main.py -c <config_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -c <config_path>')
            sys.exit()
        elif opt  == "-c":
            config_path = arg

    if config_path != '':
        with open(config_path) as json_file:
            config = json.load(json_file)
            train(config)
    else:
        print('main.py -c <config_path>')