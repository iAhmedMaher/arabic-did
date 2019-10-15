import numpy as np

class Evaluation(object):
    def __init__(self, eval_dataloader, config):
        self.dataloader = eval_dataloader
        self.metrics = config['evaluation']['metrics']
        self.device = config['device']

    def eval_model(self, model):
        print("Starting evaluation ...")
        model.eval()
        y = []
        yhat = []

        for idx, batch in enumerate(self.dataloader):
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            labels, tokens = batch
            y += list(labels.view(-1).cpu().detach().numpy())
            yhat += list(model(tokens).max(-1)[1].view(-1).cpu().detach().numpy())

        y = np.array(y)
        yhat = np.array(yhat)

        results = {}
        if 'micro_average_accuracy' in self.metrics:
            results['micro_average_accuracy'] = self.get_micro_average(y, yhat)

        model.train()
        return results

    def get_micro_average(self, y, yhat):
        return np.count_nonzero(y == yhat)/len(y)


