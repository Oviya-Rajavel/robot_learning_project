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
    
    Args:batch_size
        demos (list): List of RLBench demo objects
        max_epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        log_dir (str): Directory for TensorBoard logsepoc
    """
    # Create dataloader
    #train_loader, val_loader, test_loader = create_dataloaders(
    #    demos, 
    #    batch_size=1,  # Each batch is one full demo
    #    shuffle=True  # Optional: set to True if you want to randomize demo order
    #)
    
    train_loader, val_loader, test_loader = create_sequence_prediction_dataloaders(
        demos, 
        window_size=20,        # Number of timesteps in input sequence
        prediction_offset=1,   # Predict 1 timestep ahead
        batch_size=16          # Batch size
    )

    # Initialize model
    model = PoseTransformerLightning()
    
    # TensorBoard Logger
    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir, 
        name='pose_transformer'
    )
    
    # Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename='pose_transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],#, early_stop_callback],
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
    # Create logs directory
    os.makedirs('lightning_logs', exist_ok=True)
    
    # RLBench Environment Setup
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
    
    # Get task
    task = env.get_task(TaskboardRobothon)
    
    # Get demonstrations
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
    # print(demos[0]._observations)
    # demos = np.array(demos).flatten()

    model, trainer, logger, test_loader = train_pose_transformer(demos=demos, 
                                                                 max_epochs = max_epochs)
    
    # Save the final model
    #ckpt_save_file = "rlbench_transformer.pth"
    #torch.save(model.state_dict(), ckpt_save_file)
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