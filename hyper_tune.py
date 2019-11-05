from ray.tune import Trainable
from train import setup_training, get_optimizers, training_step
from eval import Evaluation
import torch
import os
from average import EWMA
from colorama import Fore
from colorama import Style


def replace_value_in_nested_dict(node, kv, new_value):
    if isinstance(node, list):
        for i in node:
            for x in replace_value_in_nested_dict(i, kv, new_value):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            node[kv] = new_value
            yield node[kv]
        for j in node.values():
            for x in replace_value_in_nested_dict(j, kv, new_value):
                yield x


def inject_tuned_hyperparameters(config):
    for k in config.keys():
        if k.split('|')[0] == 'allocate':
            list(replace_value_in_nested_dict(config, k.split('|')[1], config[k]))
    return config


class TuneTrainable(Trainable):
    def _setup(self, config):
        inject_tuned_hyperparameters(config)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Trainable got the following config after injection', config)
        self.config = config
        self.device = self.config['device']
        self.exp, self.model, self.train_dataloader, self.eval_dataloader, self.loss_func = setup_training(self.config)
        self.exp.set_name(config['experiment_name'] + self._experiment_id)
        self.exp.send_notification(title='Experiment ' + str(self._experiment_id) + ' ended')
        self.train_data_iter = iter(self.train_dataloader)
        self.model = self.model.to(self.device)
        self.model.train()
        self.optimizers = get_optimizers(self.model, self.config)
        self.evaluator = Evaluation(self.eval_dataloader, self.config)
        self.num_examples = 0
        self.batch_idx = 0
        self.epoch = 1
        self.ewma = EWMA(beta=0.75)
        self.last_accu = -1.0

    def get_batch(self):
        try:
            batch = next(self.train_data_iter)
            return batch

        except StopIteration:
            self.train_data_iter = iter(self.train_dataloader)
            batch = next(self.train_data_iter)
            self.batch_idx = 0
            self.epoch += 1
            return batch

    def _train(self):
        total_log_step_loss = 0
        total_log_step_train_accu = 0

        while True:
            batch = self.get_batch()
            self.batch_idx += 1
            self.num_examples += len(batch[0])
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            loss, train_accu = training_step(batch, self.model, self.optimizers, self.loss_func)
            total_log_step_loss += loss.cpu().detach().numpy()
            total_log_step_train_accu += train_accu

            if self.batch_idx % self.config['training']['log_every_n_batches'] == 0:
                avg_loss = total_log_step_loss / self.config['training']['log_every_n_batches']
                avg_accu = total_log_step_train_accu / self.config['training']['log_every_n_batches']
                print(f'{Fore.YELLOW}Total number of seen examples:', self.num_examples, 'Average loss of current log step:',
                      avg_loss, 'Average train accuracy of current log step:', avg_accu, f"{Style.RESET_ALL}")
                self.exp.log_metric('train_loss', avg_loss, step=self.num_examples, epoch=self.epoch)
                self.exp.log_metric('train_accuracy', avg_accu, step=self.num_examples, epoch=self.epoch)
                total_log_step_loss = 0
                total_log_step_train_accu = 0

            if (self.batch_idx + 1) % self.config['training']['eval_every_n_batches'] == 0:
                results, assets, image_fns = self.evaluator.eval_model(self.model, self.loss_func)
                print(self.config['tune']['discriminating_metric'], results[self.config['tune']['discriminating_metric']])
                self.exp.log_metrics(results, step=self.num_examples, epoch=self.epoch)
                [self.exp.log_asset_data(asset, step=self.num_examples) for asset in assets]
                [self.exp.log_image(fn, step=self.num_examples) for fn in image_fns]

                accu_diff_avg = abs(results[self.config['tune']['discriminating_metric']] - self.ewma.get())
                accu_diff_cons = abs(results[self.config['tune']['discriminating_metric']] - self.last_accu)

                no_change_in_accu = 1 if accu_diff_avg < 0.0005 and accu_diff_cons < 0.002 else 0
                self.ewma.update(results[self.config['tune']['discriminating_metric']])
                self.last_accu = results[self.config['tune']['discriminating_metric']]

                training_results = {
                    self.config['tune']['discriminating_metric']: results[self.config['tune']['discriminating_metric']],
                    'num_examples': self.num_examples, 'no_change_in_accu' : no_change_in_accu}

                return training_results

    def _save(self, checkpoint_dir):
        save_dict = {'model_state_dict': self.model.state_dict()}
        for i, optimizer in enumerate(self.optimizers):
            save_dict['op_' + str(i) + '_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, os.path.join(checkpoint_dir, 'checkpoint_file.pt'))
        return os.path.join(checkpoint_dir, 'checkpoint_file.pt')

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(checkpoint['op_' + str(i) + '_state_dict'])

    def stop(self):
        results, assets, image_fns = self.evaluator.eval_model(self.model, self.loss_func, finished_training=True)
        self.exp.log_metrics(results, step=self.num_examples, epoch=self.epoch)
        [self.exp.log_asset_data(asset, step=self.num_examples) for asset in assets]
        [self.exp.log_image(fn, step=self.num_examples) for fn in image_fns]
        return super().stop()
