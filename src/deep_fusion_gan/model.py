import os.path
from typing import Tuple, List

from utils import save_gen_weights_to_gdrive, save_gen_losses_to_gdrive, save_training_times_to_gdrive, save_model_to_gdrive
import pickle
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import trange

from discriminator.model import Discriminator
from generator.model import Generator
from objects.utils import prepare_data
from text_encoder.model import RNNEncoder
# from src.discriminator.model import Discriminator
# from src.generator.model import Generator
# from src.objects.utils import prepare_data
# from src.text_encoder.model import RNNEncoder


class DeepFusionGAN:
    def __init__(self, n_words, encoder_weights_path: str, image_save_path: str, gen_path_save: str, loss_path_save: str, training_time_path_save: str, model_path_save: str):
        super().__init__()
        self.image_save_path = image_save_path
        self.gen_path_save = gen_path_save
        self.loss_path_save = loss_path_save
        self.model_path_save = model_path_save
        self.training_time_path_save = training_time_path_save

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(n_channels=32, latent_dim=100).to(self.device)
        self.discriminator = Discriminator(n_c=32).to(self.device)

        self.text_encoder = RNNEncoder.load(encoder_weights_path, n_words)
        self.text_encoder.to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))

        self.relu = nn.ReLU()

    def _zero_grad(self):
        self.d_optim.zero_grad()
        self.g_optim.zero_grad()

    def _compute_gp(self, images: Tensor, sentence_embeds: Tensor) -> Tensor:
        batch_size = images.shape[0]

        images_interpolated = images.data.requires_grad_()
        sentences_interpolated = sentence_embeds.data.requires_grad_()

        embeds = self.discriminator.build_embeds(images_interpolated)
        logits = self.discriminator.get_logits(embeds, sentences_interpolated)

        grad_outputs = torch.ones_like(logits)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=(images_interpolated, sentences_interpolated),
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True
        )

        grad_0 = grads[0].reshape(batch_size, -1)
        grad_1 = grads[1].reshape(batch_size, -1)

        grad = torch.cat((grad_0, grad_1), dim=1)
        grad_norm = grad.norm(2, 1)

        return grad_norm

    def fit(self, train_loader: DataLoader, num_epochs: int = 600, checkpoint_state: dict = None) -> Tuple[List[float], List[float], List[float]]:
        g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = [], [], []
        training_times = []
        epoch_current = -1
        if (type(checkpoint_state) is dict):
            g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = checkpoint_state['g_losses_epoch'], checkpoint_state['d_losses_epoch'], checkpoint_state['d_gp_losses_epoch']
            training_times = checkpoint_state['training_times']
            epoch_current = checkpoint_state['epoch_id']
            
            self.generator.load_state_dict(checkpoint_state['model']['netG'])
            self.discriminator.load_state_dict(checkpoint_state['model']['netD'])
            self.g_optim.load_state_dict(checkpoint_state['optimizers']['g_optim'])
            self.d_optim.load_state_dict(checkpoint_state['optimizers']['d_optim'])
            
        
        for epoch in trange(num_epochs, desc="Train Deep Fusion GAN"):
            if epoch_current >= epoch:
                continue
            
            g_losses, d_losses, d_gp_losses = [], [], []
            batch_size_for_state_save = None
            start_time = time.time()
            for batch in train_loader:
                images, captions, captions_len, _ = prepare_data(batch, self.device)
                batch_size = images.shape[0]
                batch_size_for_state_save = batch_size
                
                if batch_size != checkpoint_state['batch_size']:
                    print('ERROR!!! Not same batch size last checkpoint\nIn previous checkpoint, batch size is ' + str(checkpoint_state['batch_size']) + ', but get batch size ' + batch_size)
                    return g_losses_epoch, d_losses_epoch, d_gp_losses_epoch

                sentence_embeds = self.text_encoder(captions, captions_len).detach()

                real_embeds = self.discriminator.build_embeds(images)
                real_logits = self.discriminator.get_logits(real_embeds, sentence_embeds)
                d_loss_real = self.relu(1.0 - real_logits).mean()

                shift_embeds = real_embeds[:(batch_size - 1)]
                shift_sentence_embeds = sentence_embeds[1:batch_size]
                shift_real_image_embeds = self.discriminator.get_logits(shift_embeds, shift_sentence_embeds)

                d_loss_mismatch = self.relu(1.0 + shift_real_image_embeds).mean()

                noise = torch.randn(batch_size, 100).to(self.device)
                fake_images = self.generator(noise, sentence_embeds)

                fake_embeds = self.discriminator.build_embeds(fake_images.detach())
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)

                d_loss_fake = self.relu(1.0 + fake_logits).mean()

                d_loss = d_loss_real + (d_loss_fake + d_loss_mismatch) / 2.0

                self._zero_grad()
                d_loss.backward()
                self.d_optim.step()

                d_losses.append(d_loss.item())

                grad_l2norm = self._compute_gp(images, sentence_embeds)
                d_loss_gp = 2.0 * torch.mean(grad_l2norm ** 6)

                self._zero_grad()
                d_loss_gp.backward()
                self.d_optim.step()

                d_gp_losses.append(d_loss_gp.item())

                fake_embeds = self.discriminator.build_embeds(fake_images)
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)
                g_loss = -fake_logits.mean()

                self._zero_grad()
                g_loss.backward()
                self.g_optim.step()

                g_losses.append(g_loss.item())
            end_time = time.time()

            g_losses_epoch.append(np.mean(g_losses))
            d_losses_epoch.append(np.mean(d_losses))
            d_gp_losses_epoch.append(np.mean(d_gp_losses))
            training_times.append(end_time - start_time)

            self._save_fake_image(fake_images, epoch)
            # self._save_gen_weights(epoch)
            # self._save_losses_epoch(epoch, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch)
            # self._save_trainings_time_epoch(epoch, training_times)
            self._save_model(epoch, batch_size_for_state_save, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch, training_times)
            # if (epoch + 1) % 10 == 0:
            #     self._save_fake_image(fake_images, epoch)
            #     self._save_gen_weights(epoch)

        return g_losses_epoch, d_losses_epoch, d_gp_losses_epoch

    def _save_fake_image(self, fake_images: Tensor, epoch: int):
        img_path = os.path.join(self.image_save_path, f"fake_sample_epoch_{epoch}.png")
        vutils.save_image(fake_images.data, img_path, normalize=True)

    def _save_gen_weights(self, epoch: int):
        gen_path = os.path.join(self.gen_path_save, f"gen_{epoch}.pth")
        torch.save(self.generator.state_dict(), gen_path)
        
        save_gen_weights_to_gdrive(gen_path)
        
    def _save_model(self, epoch: int, batch_size, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch, training_times):
        model_save_path = os.path.join(self.model_path_save, f"model_checkpoint_{epoch}.pth")
        state = {'model': {'netG': self.generator.state_dict(), 'netD': self.discriminator.state_dict()}, \
                'optimizers': {'g_optim': self.g_optim.state_dict(), 'd_optim': self.d_optim.state_dict()}, \
                'epoch_id': epoch, \
                'batch_size': batch_size, \
                'g_losses_epoch': g_losses_epoch, \
                'd_losses_epoch': d_losses_epoch, \
                'd_gp_losses_epoch': d_gp_losses_epoch, \
                'training_times': training_times, \
                'Note for Information detail': 'epoch_id from 0 --> n-1, EX: N = 600 epoches --> epoch_id from 0 --> 599'}
        # 'device': self.device, \
        torch.save(state, model_save_path)
        
        save_model_to_gdrive(model_save_path)
    
    def _save_losses_epoch(self, epoch:int, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch):
        # Note We save list loss, from epoch 1 --> current epoch
        loss_path_g = os.path.join(self.loss_path_save, f"g_losses_epoch_{epoch}.pth")
        loss_path_d = os.path.join(self.loss_path_save, f"d_losses_epoch_{epoch}.pth")
        loss_path_d_gp = os.path.join(self.loss_path_save, f"d_gp_losses_epoch_{epoch}.pth")
        
        with open(loss_path_g, 'wb') as file:
            pickle.dump(g_losses_epoch, file)
        with open(loss_path_d, 'wb') as file:
            pickle.dump(d_losses_epoch, file)
        with open(loss_path_d_gp, 'wb') as file:
            pickle.dump(d_gp_losses_epoch, file)
            
        save_gen_losses_to_gdrive(loss_path_g)
        save_gen_losses_to_gdrive(loss_path_d)
        save_gen_losses_to_gdrive(loss_path_d_gp)
        
    def _save_trainings_time_epoch(self, epoch:int, training_times):
        # Note We save list training times, from epoch 1 --> current epoch
        training_time_path = os.path.join(self.training_time_path_save, f"training_times_{epoch}.pth")
        
        with open(training_time_path, 'wb') as file:
            pickle.dump(training_times, file)
        
        save_training_times_to_gdrive(training_time_path)
