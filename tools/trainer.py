from math import fabs
import torch
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler as optim_lr
import torch.nn as nn

from .early_stopping import EarlyStopping
from .model_utils import get_trainable_layers
from .model_utils import nll
from .parts import SAVED_OUTPUT_NAME, SAVED_INPUT_NAME, PartManager
from .reconstructions import batchnorm_inverse, conv_inverse, fc_inverse, conv_inverse_channelwise
from .config import Config

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

    
class ModelTrainer:
    def __init__(self, model : torch.nn.Module, cfg : Config, part_manager : PartManager):
        self.model = model
        self.config = cfg
        self.part_manager = part_manager
        self.train_losses = []
        self.train_counter = []
        self.early_stopping = not hasattr(cfg, 'early_stopping') or cfg.early_stopping
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        print(f"{self.config.model_name} to {self.device}")
        
        self.model.to(self.device)
    
    def load_model(self):
            model_path = os.path.join(self.config.checkpoint_path, f'{self.config.model_name}.pth')
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
            
    def set_aux_layer(self, part, size_multiplier=3):
        part_conv_layer = part.get_conv_layer()
        out_channels, in_channels, hk, wk = part_conv_layer.weight.shape
        middle_channels = out_channels*size_multiplier
        self.model.aux_conv = nn.Conv2d(in_channels, middle_channels, kernel_size=hk, bias=True).to(self.device)
        self.model.aux_reduce_conv = nn.Conv2d(middle_channels, out_channels, kernel_size=1, bias=True).to(self.device)
        print(f"Aux part in_channels: {in_channels}, middle_channels: {middle_channels}, out_channels: {out_channels}")
        
    def get_model_aux_output(self, x):
        x = self.model.aux_conv(x)
        x = self.model.aux_reduce_conv(x)
        return x
    
    def model_aux_loss(self, part):
        x_part = self.get_part_input(part)
        y_part = self.get_part_output(part)
        y_aux = self.get_model_aux_output(x_part)
        return torch.sqrt(torch.mean(torch.square(y_aux - y_part)))
        
    def train(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader = None):
        self.optimizer = None
        self.lr_scheduler = None
        self.es = EarlyStopping() if self.early_stopping else None
        try:
            optim_params = self.config.optimizer.copy()
            optim_name = optim_params.pop("name")
            self.optimizer = getattr(optim, optim_name)(get_trainable_layers(self.model), **optim_params)
        except:
            print("No optimizer set.")
            
        try:
            scheduler_params = self.config.lr_scheduler.copy()
            scheduler_name = scheduler_params.pop("name")
            self.lr_scheduler = getattr(optim_lr, scheduler_name)(self.optimizer, **scheduler_params)
        except:
            print("No scheduler set.")
        
        best_val_loss = np.inf
        for epoch in range(1, self.config.n_epochs + 1):
            self.model.train()
            
            # Training
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(X_batch.to(self.device))
                
                loss = None
                if self.config.loss_fn == 'nll':
                    loss = nll(output, y_batch.to(self.device))
                    
                    for i_part in range(self.part_manager.train_part_i + 1):
                        loss += self.part_reconstruction_loss(self.part_manager.parts[i_part]) * self.config.part_reconstruction_loss_multiplier
                elif self.config.loss_fn == 'aux':
                    i_part = self.part_manager.train_part_i
                    loss = self.model_aux_loss(self.part_manager.parts[i_part])
                    
                else:
                    raise AttributeError("config.trainer.loss_fn is invalid.")

                loss.backward()
                self.optimizer.step()
                if not self.config.log_interval is None and batch_idx % self.config.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tLR:{:.6f}'.format(
                        epoch, batch_idx * len(X_batch), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), self.optimizer.param_groups[0].get('lr')))
                    
                    self.train_losses.append(loss.item())
                    self.train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            
            # Validation
            if val_loader is not None:
                val_loss = 0
                n_total_examples = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        n_examples = X_batch.size()[0]
                        n_total_examples += n_examples
                        output = self.model(X_batch.to(self.device))
                        
                        loss = None
                        if self.config.loss_fn == 'nll':
                            loss = nll(output, y_batch.to(self.device))
                            
                            for i_part in range(self.part_manager.train_part_i + 1):
                                loss += self.part_reconstruction_loss(self.part_manager.parts[i_part]) * self.config.part_reconstruction_loss_multiplier
                        elif self.config.loss_fn == 'aux':
                            i_part = self.part_manager.train_part_i
                            loss = self.model_aux_loss(self.part_manager.parts[i_part])
                        val_loss += loss * n_examples
                    
                    val_loss = val_loss / n_total_examples
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model()
                    print(f"Val Loss: {val_loss} (Best: {best_val_loss})")
                    
                    if not self.es is None and self.es.step(val_loss):
                        break
            else:
                self.save_model()
            
            if not self.lr_scheduler is None:
                print("lr_scheduler step")
                self.lr_scheduler.step()
                    
                    
        if val_loader is not None:
            self.load_checkpoint()
        #torch.save(self.optimizer.state_dict(), os.path.join(self.config.checkpoint_path, 'optimizer.pth'))

    def save_model(self):
        model_save_path = os.path.join(self.config.checkpoint_path, f'{self.config.model_name}.pth') # _{self.model.n_active_layers}
        print(f"Saving model to {model_save_path}")
        torch.save(self.model.state_dict(), model_save_path)
        
    def load_checkpoint(self):
        model_save_path = os.path.join(self.config.checkpoint_path, f'{self.config.model_name}.pth') # _{self.model.n_active_layers}
        print(f"Loading checkpoint from {model_save_path}")
        self.model.load_state_dict(torch.load(model_save_path))
    
    def eval_reconstruction_loss(self, data_loader : torch.utils.data.DataLoader):
        self.model.eval()
        losses = []
        n_total_examples = 0
        with torch.no_grad():
            for data, target in data_loader:
                n_examples = data.size()[0]
                n_total_examples += n_examples
                output = self.model(data.to(self.device))
                loss = self.part_reconstruction_loss(self.part_manager.parts[self.part_manager.train_part_i]).to('cpu').data
                losses.append(loss * n_examples)
        
        if n_total_examples == 0:
            raise ValueError("n_total_examples is 0")
        return np.sum(losses) / n_total_examples
                
    def test(self, data_loader : torch.utils.data.DataLoader):
        preds = []
        labels = []
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data.to(self.device))
                pred = output.to('cpu').data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

                preds.extend(list(pred.numpy()))
                labels.extend(list(target.numpy()))
        print('\nTest set accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

        print(classification_report(labels, preds))
        return confusion_matrix(labels, preds)
    
    def test_accuracy(self, test_loader):
        preds = []
        labels = []
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data.to(self.device))
                pred = output.to('cpu').data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

                preds.extend(list(pred.numpy()))
                labels.extend(list(target.numpy()))
                
        return (correct / len(test_loader.dataset)).numpy().item()
    
    def get_part_input(self, part):
        return getattr(part.get_conv_layer(), SAVED_INPUT_NAME).clone()
    
    def get_part_output(self, part):
        return getattr(part.get_conv_layer(), SAVED_OUTPUT_NAME).clone()
    
    def get_reconstruction(seif, part):
        rec = getattr(part.get_conv_layer(), SAVED_OUTPUT_NAME).clone()
        
        # A typical part in resnet has 3 layers:
        # - pooling (optional)
        # - conv
        # - batchnorm
        # - relu (optional)
        
        if not part.has_relu:
            rec[rec < 0] = 0.0
            
        if part.has_bn:
            rec = batchnorm_inverse(rec, part.get_bn_layer())
        
        if part.has_conv:
            conv_layer = part.get_conv_layer()
            rec = conv_inverse(rec, conv_layer)
        else:
            if not part.has_linear:
                raise AttributeError("A part must have a linear or conv layer.")
            rec = fc_inverse(rec, part.get_linear_layer())
            
        # Assuming activation is >= 0
        if part.index > 0:
            rec[rec < 0] = 0.0
        return rec
            
    def get_input_to_reconstruct(self, part):
        return getattr(part.get_loss_start_layer(), SAVED_INPUT_NAME)
            
    def part_reconstruction_loss(self, part):
        rec = self.get_reconstruction(part).clone()
        to_reconstruct = self.get_input_to_reconstruct(part).clone()
            
        #rec_error = torch.square(rec - to_reconstruct)
        #loss = torch.sqrt(torch.sum(rec_error) / to_reconstruct.size()[0]) / torch.sqrt(torch.mean(torch.square(to_reconstruct)))# Divide by batch size
        loss = torch_cos_similarity_loss(rec, to_reconstruct)
        return loss
    
    def part_activation_loss(self, part):
        activations = getattr(part.get_loss_end_layer(), SAVED_OUTPUT_NAME)
        return activation_loss(activations)
    
    def part_gram_matrix_loss(self, part):
        activations = getattr(part.get_loss_end_layer(), SAVED_OUTPUT_NAME)
        return gram_matrix_loss(activations)
    
    def get_reconstruction_channelwise(seif, part):
        rec = getattr(part.get_loss_end_layer(), SAVED_OUTPUT_NAME).clone()
        
        # A typical part in resnet has 3 layers:
        # - pooling (optional)
        # - conv
        # - batchnorm
        # - relu (optional)
        
        if not part.has_relu:
            rec[rec < 0] = 0.0
            
        if part.has_bn:
            rec = batchnorm_inverse(rec, part.get_bn_layer())
        
        if part.has_conv:
            conv_layer = part.get_conv_layer()
            rec = conv_inverse(rec, conv_layer)
        else:
            if not part.has_linear:
                raise AttributeError("A part must have a linear or conv layer.")
            rec = fc_inverse(rec, part.get_linear_layer())
            
        # Assuming activation is >= 0
        if part.index > 0:
            rec[rec < 0] = 0.0
        return rec
    
    def get_channel_reconstruction(self, part, i_channel):
        rec = getattr(part.get_loss_end_layer(), SAVED_OUTPUT_NAME).clone()
        conv_layer = part.get_conv_layer()
        rec_channel = rec.clone()[:, i_channel, :, :]
        rec_channel = rec_channel.unsqueeze(1)
        
        rec_channel = conv_inverse_channelwise(rec_channel, conv_layer, i_channel)
        
        # Assuming activation is >= 0
        if part.index > 0:
            rec_channel[rec_channel < 0] = 0.0
        return rec_channel
        
    def part_reconstruction_loss_channelwise(self, part):
        if not part.has_conv:
            return self.part_reconstruction_loss(part)
        channel_losses = []
        to_reconstruct = self.get_input_to_reconstruct(part).clone()
        total_channel_loss = 0.0
        rec = getattr(part.get_loss_end_layer(), SAVED_OUTPUT_NAME).clone()
        conv_layer = part.get_conv_layer()
        n_channels = rec.shape[1]
        #print(f'{n_channels=}')
        for i_channel in range(n_channels):
            #print(f'{rec.shape=}')
            rec_channel = rec.clone()[:, i_channel, :, :]
            rec_channel = rec_channel.unsqueeze(1)
            
            rec_channel = conv_inverse_channelwise(rec_channel, conv_layer, i_channel)
            
            # Assuming activation is >= 0
            if part.index > 0:
                rec_channel[rec_channel < 0] = 0.0
            
            channel_loss = torch_cos_similarity_loss(rec_channel, to_reconstruct)
            #print(f'channel_loss:{channel_loss}')
            total_channel_loss += channel_loss
            channel_losses.append(channel_loss)
        
        self.channel_losses = [c.cpu().detach().numpy().item() for c in channel_losses]
        return (total_channel_loss / n_channels)

def torch_cos_similarity_loss(x1, x2):
    #return - torch.log(F.cosine_similarity(x1.flatten(start_dim=2), x2.flatten(start_dim=2), dim=2, eps=1e-5) + 1e-3).mean(axis=1).mean()
    return 1 - F.cosine_similarity(x1.flatten(start_dim=2), x2.flatten(start_dim=2), dim=2, eps=1e-5).mean()

def activation_loss(activations):
    flat_act = activations.flatten(start_dim=2)
    #loss = - torch.log(1 - ((flat_act.max(dim=2).values - flat_act.mean(dim=2)).mean() / 10))
    loss = - (flat_act.max(dim=2).values - flat_act.mean(dim=2) * 10).mean()
    return loss

def gram_matrix_mean(gen):
    batch_size,channel,height,width=gen.shape
    mean_vals = torch.zeros(batch_size)
    for b in range(batch_size):
        gen_batch = gen[b]
        G=torch.mm(gen_batch.view(channel,height*width),gen_batch.view(channel,height*width).t())
        mean_vals[b] = torch.abs(G).mean() / torch.mean(torch.square(gen_batch.view(channel,height*width)), axis=1).mean()
    return mean_vals.mean()

def gram_matrix_loss(activations):
    loss = gram_matrix_mean(activations)
    return loss