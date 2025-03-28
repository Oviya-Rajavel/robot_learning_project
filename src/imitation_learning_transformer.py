import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks.taskboard_robothon import TaskboardRobothon
# from behaviour_cloning_transformer import PoseTransformerLightning
from behaviour_cloning_dataloader import create_dataloaders
from sequence_dataloader import SequencePredictionDataset, create_sequence_prediction_dataloaders
# from behaviour_clonning_sequence import PoseTransformerLightning
from final_behaviour_cloning import PoseTransformerLightning

def train_pose_transformer(demos, 
                            max_epochs=50, 
                            batch_size=1, 
                            log_dir='./lightning_logs_window_final',
                            train = True):
    """
    Train the Pose Transformer using PyTorch Lightning
    """
    
    train_loader, val_loader, test_loader = create_sequence_prediction_dataloaders(
        demos, 
        window_size=20,       
        prediction_offset=1,   
        batch_size=16         
    )

    model = PoseTransformerLightning()

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir, 
        name='pose_transformer'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename='pose_transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    

    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        check_val_every_n_epoch=1
    )
    
    # Train the model
    if train:
        trainer.fit(
            model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )
    
    return model, trainer, logger, test_loader

def main():
    os.makedirs('lightning_logs', exist_ok=True)
    

    cam_config = CameraConfig(mask=False)
    obs_config = ObservationConfig(
        left_shoulder_camera=cam_config,
        right_shoulder_camera=cam_config,
        overhead_camera=cam_config,
        wrist_camera=cam_config,
        front_camera=cam_config
    )
    
    # Set up environment
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), 
            gripper_action_mode=Discrete()
        ),
        dataset_root='/media/sun/Expansion/Robot_learning',  # Update with your dataset path
        obs_config=obs_config,
        headless=True
    )
    env.launch()

    task = env.get_task(TaskboardRobothon)

    live_demos = False
    all_demos = True
    max_epochs = 1
    print("Loading demos....")
    if all_demos:
        max_epochs = 50
        demos = task.get_demos(-1, live_demos=live_demos)
    else:
        max_epochs = 1
        demos = task.get_demos(1, live_demos=live_demos)
    print("Demo loading completed....")


    model, trainer, logger, test_loader = train_pose_transformer(demos=demos, 
                                                                 max_epochs = max_epochs)
    

    ckpt_save_file = "rlbench_transformer_window.ckpt"
    trainer.save_checkpoint(ckpt_save_file)
    print("Training complete!")
    ##testing the model:
    print("Testing....")
    
    model = PoseTransformerLightning.load_from_checkpoint(ckpt_save_file)
    trainer.test(model, dataloaders=test_loader)
    # Shutdown environment
    env.shutdown()

if __name__ == "__main__":
    main()