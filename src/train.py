import os
from typing import List, Tuple

from deep_fusion_gan.model import DeepFusionGAN
from utils import create_loader, fix_seed
# from src.deep_fusion_gan.model import DeepFusionGAN
# from src.utils import create_loader, fix_seed


def train() -> Tuple[List[float], List[float], List[float]]:
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

    train_loader = create_loader(256, 24, data_path, "train")
    model = DeepFusionGAN(n_words=train_loader.dataset.n_words,
                          encoder_weights_path=encoder_weights_path,
                          image_save_path=image_save_path,
                          gen_path_save=gen_path_save,
                          loss_path_save=loss_path_save,
                          training_time_path_save=training_time_path_save,
                          model_path_save=model_path_save)

    return model.fit(train_loader)


if __name__ == '__main__':
    g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = train()
    
    print("g_losses_epoch:", g_losses_epoch)
    print("d_losses_epoch:", d_losses_epoch)
    print("d_gp_losses_epoch:", d_gp_losses_epoch)
