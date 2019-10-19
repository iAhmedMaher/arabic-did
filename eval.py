import numpy as np


class Evaluation(object):
    def __init__(self, eval_dataloader, config):
        self.dataloader = eval_dataloader
        self.metrics = config['evaluation']['metrics']
        self.device = config['device']
        self.labels_to_int = config['labels_to_int']

    def eval_model(self, model):
        print("Starting evaluation ...")
        model.eval()
        labels = []
        outputs = []

        for idx, batch in enumerate(self.dataloader):
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            labels_batch, tokens_batch = batch
            labels += [labels_batch]
            outputs += [model(tokens_batch)]

        cm = self.get_confusion_matrix(labels, outputs)

        results = {}

        if 'per_class_precision' in self.metrics:
            results.update(self.get_per_class_precision(cm))
        if 'per_class_recall' in self.metrics:
            results.update(self.get_per_class_recall(cm))
        if 'per_class_f1' in self.metrics:
            results.update(self.get_per_class_f1(cm))
        if 'micro_average_accuracy' in self.metrics:
            results.update(self.get_micro_average_accuracy(cm))
        if 'macro_average_precision' in self.metrics:
            results.update(self.get_macro_precision(cm))
        if 'macro_average_recall' in self.metrics:
            results.update(self.get_macro_recall(cm))
        if 'macro_average_f1' in self.metrics:
            results.update(self.get_macro_f1(cm))

        model.train()
        return results


    def get_confusion_matrix(self, labels, outputs):
        labels_np = np.concatenate([label.cpu().detach().numpy() for label in labels], axis=0)
        output_labels_np = np.concatenate([output.max(-1)[1].view(-1).cpu().detach().numpy() for output in outputs], axis=0)

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
        for idx, precision in enumerate(per_class_precision):
            # TODO
            result[list(self.labels_to_int.keys())[idx] + '_precision'] = precision

        return result

    def get_per_class_recall(self, confusion_matrix):
        per_class_recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        result = {}
        for idx, recall in enumerate(per_class_recall):
            # TODO
            result[list(self.labels_to_int.keys())[idx] + '_recall'] = recall

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

    # This is also micro precision, micro recall, and micro f1 in case of multi-class classifcation
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








