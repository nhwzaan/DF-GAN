import os
from typing import List, Tuple

import torch
import argparse

from deep_fusion_gan.model import DeepFusionGAN
from utils import create_loader, fix_seed
# from src.deep_fusion_gan.model import DeepFusionGAN
# from src.utils import create_loader, fix_seed


def parse_args():
    parser = argparse.ArgumentParser(description='ResumeTrainingTex2Image')
    parser.add_argument('--checkpoint', type=str, default='../model_checkpoints/model_checkpoint_0.pth',
                        help='Path to your saved checkpoint')
    parser.add_argument('--numEpochs', type=int, default=600,
                        help='Numbers of Epochs (default: 600)')
    
    args = parser.parse_args()
    return args


def train(args) -> Tuple[List[float], List[float], List[float]]:
    fix_seed()
    
    current_working_dir = os.getcwd()
    data_path = os.path.join(current_working_dir, "data")
    encoder_weights_path = os.path.join(current_working_dir, "text_encoder_weights/text_encoder200.pth")
    image_save_path = os.path.join(current_working_dir, "gen_images")
    gen_path_save = os.path.join(current_working_dir, "gen_weights")
    loss_path_save = os.path.join(current_working_dir, "gen_losses")
    training_time_path_save = os.path.join(current_working_dir, "training_times")
    model_path_save=os.path.join(current_working_dir, "model_checkpoints")

    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(gen_path_save, exist_ok=True)
    os.makedirs(loss_path_save, exist_ok=True)
    os.makedirs(training_time_path_save, exist_ok=True)
    os.makedirs(model_path_save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(args.checkpoint, map_location=device)
    batch_size=checkpoint['batch_size']
    train_loader = create_loader(256, batch_size, data_path, "train")
    model = DeepFusionGAN(n_words=train_loader.dataset.n_words,
                          encoder_weights_path=encoder_weights_path,
                          image_save_path=image_save_path,
                          gen_path_save=gen_path_save,
                          loss_path_save=loss_path_save,
                          training_time_path_save=training_time_path_save,
                          model_path_save=model_path_save)

    return model.fit(train_loader=train_loader, num_epochs=args.numEpochs, checkpoint_state=checkpoint)


if __name__ == '__main__':
    args = parse_args()
    
    g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = train(args)
    
    print("g_losses_epoch:", g_losses_epoch)
    print("d_losses_epoch:", d_losses_epoch)
    print("d_gp_losses_epoch:", d_gp_losses_epoch)
