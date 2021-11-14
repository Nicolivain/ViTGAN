import datetime
import os

import torch
import torch.nn as nn


class PytorchNN(nn.Module):
    def __init__(self, criterion, logger=None, opt='adam', device='cpu', ckpt_save_path=None):
        super().__init__()
        self.opt_type = opt if opt in ['sgd', 'adam'] else ValueError
        self.opt = None
        self.log = logger
        self.device = device
        self.ckpt_save_path = ckpt_save_path
        self.state = {}
        self.criterion = nn.MSELoss() if criterion == 'mse' else nn.CrossEntropyLoss() if criterion == 'cross-entropy' else criterion

        self.best_criterion = {'loss' : 10**10,  'v_loss': 10**10, 'acc': -1, 'v_acc': -1}
        self.best_model = None
        self.best_epoch = None

    def _train_epoch(self, dataloader):
        epoch_loss = 0
        epoch_acc  = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).reshape(-1).long() if type(self.criterion) == nn.CrossEntropyLoss else batch_y.to(self.device)
            batch_output = self(batch_x)
            if type(self.criterion) == nn.CrossEntropyLoss:
                batch_output = batch_output.reshape(-1, batch_output.shape[-1])  # Pour time series
                n_correct = (torch.argmax(batch_output, dim=-1) == batch_y).sum().item()
                epoch_acc += n_correct/batch_output.shape[0]
            loss = self.criterion(batch_output, batch_y)
            loss.backward()
            epoch_loss += loss.item()
            self.opt.step()
            self.opt.zero_grad()
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

    def _validate(self, dataloader):
        epoch_loss = 0
        epoch_acc  = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).reshape(-1).long() if type(self.criterion) == nn.CrossEntropyLoss else batch_y.to(self.device)
            batch_output = self(batch_x)
            batch_output = batch_output.reshape(-1, batch_output.shape[-1])  # reshape in case of input of size (L, B, F)
            if type(self.criterion) == nn.CrossEntropyLoss:
                n_correct = (torch.argmax(batch_output, dim=1) == batch_y).sum().item()
                epoch_acc += n_correct / batch_x.shape[0]
            loss = self.criterion(batch_output, batch_y)
            epoch_loss += loss.item()
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

    def fit(self, dataloader, n_epochs, lr, validation_data=None, verbose=1, save_criterion='loss', ckpt=None):
        if self.opt_type == 'sgd':
            self.opt = torch.optim.SGD(params=self.parameters(), lr=lr)
        elif self.opt_type == 'adam':
            self.opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        else:
            raise ValueError('Unknown optimizer')

        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        start_epoch = 0
        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']

        for n in range(start_epoch, n_epochs):
            train_loss, train_acc = self._train_epoch(dataloader)
            val_loss, val_acc = 0, 0
            if validation_data is not None:
                with torch.no_grad():
                    val_loss, val_acc = self._validate(validation_data)

            epoch_result = {'loss': train_loss, 'acc': train_acc, 'v_loss': val_loss, 'v_acc': val_acc}
            if self.log:
                self.log('Train loss', train_loss, n)
                self.log('Val loss', val_loss, n)
                if type(self.criterion) == nn.CrossEntropyLoss:
                    self.log('Train acc', train_acc, n)
                    self.log('Val acc', val_acc, n)

            if 'acc' in save_criterion:
                if epoch_result[save_criterion] >= self.best_criterion[save_criterion]:
                    self.best_criterion = epoch_result
                    self.__save_state(n)
            elif 'loss' in save_criterion:
                if epoch_result[save_criterion] <= self.best_criterion[save_criterion]:
                    self.best_criterion = epoch_result
                    self.__save_state(n)

            if n % verbose == 0:
                if type(self.criterion) == nn.CrossEntropyLoss:
                    print('Epoch {:3d} loss: {:1.4f} Validation loss: {:1.4f} Train acc: {:1.4f} Validation acc: {:1.4f} | Best epoch {:3d}'.format(n, train_loss, val_loss, train_acc, val_acc, self.best_epoch))
                else:
                    print('Epoch {:3d} loss: {:1.4f} Validation loss: {:1.4f} | Best epoch {:3d}'.format(n, train_loss, val_loss, self.best_epoch))

            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = n
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{n}.ckpt'))

    def save(self, lr, n):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.state_dict()
        if not os.path.exists(self.ckpt_save_path):
            os.mkdir(self.ckpt_save_path)
        torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{self.start_time}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])

    def __save_state(self, n):
        self.best_epoch = n
        self.best_model = self.state_dict()

    def __load_saved_state(self):
        if self.best_model is None:
            raise ValueError('No saved model available')
        self.load_state_dict(self.best_model)
