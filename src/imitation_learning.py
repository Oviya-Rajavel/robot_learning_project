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
from behaviour_cloning_cnn import ImitationLearningCNN, RLBenchDemoDataset


def main():

    os.makedirs('logs', exist_ok=True)

    cam_config = CameraConfig(mask=False)
    obs_config = ObservationConfig(
        left_shoulder_camera=cam_config,
        right_shoulder_camera=cam_config,
        overhead_camera=cam_config,
        wrist_camera=cam_config,
        front_camera=cam_config
    )

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), 
            gripper_action_mode=Discrete()
        ),
        dataset_root='/media/sun/Expansion/Robot_learning',  # dataset path
        obs_config=obs_config,
        headless=True
    )
    env.launch()
    

    task = env.get_task(TaskboardRobothon)
    

    live_demos = False
    demos = task.get_demos(-1, live_demos=live_demos)
    

    dataset = RLBenchDemoDataset(demos)

    total_size = len(dataset)

    train_ratio=0.8
    val_ratio=0.15

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #len(train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False, 
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir='logs',
        name='rlbench_imitation_learning',
        version=None
    )

    model = ImitationLearningCNN(input_channels=3) #, pose_dim=7
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=[0],
        deterministic=True,
        log_every_n_steps=1,
        logger=tensorboard_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath='logs/checkpoints',
                filename='rlbench_imitation_learning-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss'
            )
        ]
    )
    
    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    ckpt_save_file = "rlbench_cnn_single_img.ckpt"
    trainer.save_checkpoint(ckpt_save_file)
    print("Training complete!")
    ##testing the model:
    print("Testing....")
    
    model = ImitationLearningCNN.load_from_checkpoint(ckpt_save_file)
    trainer.test(model, dataloaders=test_loader)
    
    # Shutdown environment
    env.shutdown()
    
    print("Training complete!")

if __name__ == "__main__":
    main()