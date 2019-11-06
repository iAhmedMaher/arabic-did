import numpy as np
import json
import pandas as pd
from cm_plotter import pretty_plot_confusion_matrix
import uuid
from tqdm import tqdm
import os


class Evaluation(object):
    def __init__(self, eval_dataloader, config):
        self.dataloader = eval_dataloader
        self.metrics = config['evaluation']['metrics']
        self.device = config['device']
        self.labels_to_int = config['labels_to_int']

    def eval_model(self, model, finished_training=False):
        print("Starting evaluation ...")
        model.eval()
        labels = []
        predicted_labels = []
        eval_losses = []
        original_texts = []
        processed_texts = []

        for idx, batch in enumerate(self.dataloader):
            if finished_training:
                original_texts += batch[2]
                processed_texts += batch[3]

            batch = (batch[0].to(self.device), batch[1].to(self.device))
            labels_batch, tokens_batch = batch
            predicted_labels_batch, eval_loss_batch = model(tokens_batch, labels_batch)
            eval_losses += [eval_loss_batch.cpu().detach()]
            predicted_labels += [predicted_labels_batch.cpu().detach()]
            labels += [labels_batch.cpu().detach()]

        cm = self.get_confusion_matrix(labels, predicted_labels)

        metrics = {}
        assets = []
        images_fns = []

        if 'per_class_precision' in self.metrics:
            metrics.update(self.get_per_class_precision(cm))
        if 'per_class_recall' in self.metrics:
            metrics.update(self.get_per_class_recall(cm))
        if 'per_class_f1' in self.metrics:
            metrics.update(self.get_per_class_f1(cm))
        if 'micro_average_accuracy' in self.metrics:
            metrics.update(self.get_micro_average_accuracy(cm))
        if 'macro_average_precision' in self.metrics:
            metrics.update(self.get_macro_precision(cm))
        if 'macro_average_recall' in self.metrics:
            metrics.update(self.get_macro_recall(cm))
        if 'macro_average_f1' in self.metrics:
            metrics.update(self.get_macro_f1(cm))
        if 'eval_loss' in self.metrics:
            metrics.update({'eval_loss' : np.mean(np.array(eval_losses))})
        if 'in_out' in self.metrics and finished_training:
            assets += self.get_text_y_yhat(predicted_labels, labels, original_texts, processed_texts)
        if 'cm' in self.metrics and finished_training:
            images_fns += self.save_and_get_cm_image(cm)

        model.train()
        return metrics, assets, images_fns


    def get_confusion_matrix(self, labels, predicted_labels):
        labels_np = np.concatenate([label.numpy() for label in labels], axis=0)
        output_labels_np = np.concatenate([predicted_label.numpy() for predicted_label in predicted_labels], axis=0)

        n_classes = len(self.labels_to_int)
        confusion_matrix = np.zeros((n_classes, n_classes))

        for row_label in self.labels_to_int.values():
            for column_label in self.labels_to_int.values():
                confusion_matrix[row_label, column_label] = np.count_nonzero(
                    output_labels_np[labels_np == row_label] == column_label)

        return confusion_matrix

    def get_per_class_precision(self, confusion_matrix):
        per_class_precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        result = {}

        for label, int_label in self.labels_to_int.items():
            result[label + '_precision'] = per_class_precision[int_label]

        return result

    def get_per_class_recall(self, confusion_matrix):
        per_class_recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        result = {}

        for label, int_label in self.labels_to_int.items():
            result[label + '_recall'] = per_class_recall[int_label]

        return result

    def get_per_class_f1(self, confusion_matrix):
        per_class_precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        per_class_recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        result = {}
        for label in self.labels_to_int.keys():
            int_label = self.labels_to_int[label]
            result['f1_' + label] = (2 * per_class_precision[int_label] * per_class_recall[int_label]) / (
                    per_class_precision[int_label] + per_class_recall[int_label])

        return result

    # This is also micro precision, micro recall, and micro f1 in case of multi-class classification
    def get_micro_average_accuracy(self, confusion_matrix):
        return {'micro_average_accuracy' : np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)}

    def get_macro_precision(self, confusion_matrix):
        return {'macro_average_precision': np.mean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))}

    def get_macro_recall(self, confusion_matrix):
        return {'macro_average_recall': np.mean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))}

    def get_macro_f1(self, confusion_matrix):
        p = np.mean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))
        r = np.mean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
        return {'macro_average_f1' : (2*p*r)/(p+r)}

    def get_text_y_yhat(self, predicted_labels, labels, original_texts, processed_texts):
        labels_np = np.concatenate([label.numpy() for label in labels], axis=0)
        output_labels_np = np.concatenate([predicted_label.numpy() for predicted_label in predicted_labels], axis=0)
        asset = []

        for idx in range(len(original_texts)):
            asset.append({'original_text' : original_texts[idx],
                          'processed_texts' : processed_texts[idx],
                          'y': list(self.labels_to_int.keys())[labels_np[idx]],
                          'yhat': list(self.labels_to_int.keys())[output_labels_np[idx]]})

        return [json.dumps(asset, ensure_ascii=False).encode('utf8')]

    def save_and_get_cm_image(self, cm):
        fname = os.path.join('dump', 'cm' + uuid.uuid4().hex + '.png')
        df_cm = pd.DataFrame(cm, list(self.labels_to_int.keys()),list(self.labels_to_int.keys()))
        pretty_plot_confusion_matrix(fname, df_cm, pred_val_axis='x')
        return [fname]




