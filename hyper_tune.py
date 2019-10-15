from ray.tune import Trainable
from train import setup_training, get_optimizers, training_step
from eval import Evaluation
import torch
import os


class TuneTrainable(Trainable):
    def _setup(self, config):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.config = config
        self.device = self.config['device']
        self.exp, self.model, self.train_dataloader, self.eval_dataloader = setup_training(self.config)
        self.exp.set_name(config['experiment_name'] + self._experiment_id)
        self.train_data_iter = iter(self.train_dataloader)
        self.model = self.model.to(self.device)
        self.model.train()
        self.optimizers = get_optimizers(self.model, self.config)
        self.evaluator = Evaluation(self.eval_dataloader, self.config)
        self.num_examples = 0
        self.batch_idx = -1
        self.epoch = 1

    def get_batch(self):
        try:
            batch = next(self.train_data_iter)
            return batch

        except StopIteration:
            self.train_data_iter = iter(self.train_dataloader)
            batch = next(self.train_data_iter)
            self.batch_idx = -1
            self.epoch += 1
            return batch


    def _train(self):
        while True:
            batch = self.get_batch()
            self.batch_idx += 1
            self.num_examples += len(batch[0])
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            loss = training_step(batch, self.model, self.optimizers)

            if self.batch_idx % self.config['training']['log_every_n_batches'] == 0:
                print(self.num_examples, loss.detach().cpu().numpy())
                self.exp.log_metric('train_loss', loss.detach().cpu().numpy(), step=self.num_examples, epoch=self.epoch)

            if (self.batch_idx + 1) % self.config['training']['eval_every_n_batches'] == 0:
                results = self.evaluator.eval_model(self.model)
                for metric in results:
                    print(metric, results[metric])
                    self.exp.log_metric(metric, results[metric], step=self.num_examples, epoch=self.epoch)

                training_results = {
                    self.config['tune']['discriminating_metric']: results[self.config['tune']['discriminating_metric']],
                    'num_examples': self.num_examples}

                return training_results

    def _save(self, checkpoint_dir):
        save_dict = {'model_state_dict': self.model.state_dict()}
        for i, optimizer in enumerate(self.optimizers):
            save_dict['op_' + str(i) + '_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, os.path.join(checkpoint_dir, 'checkpoint_file.pt'))

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(checkpoint['op_' + str(i) + '_state_dict'])
