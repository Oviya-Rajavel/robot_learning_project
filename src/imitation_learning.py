# import numpy as np

# from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rlbench.action_modes.arm_action_modes import JointVelocity
# from rlbench.action_modes.gripper_action_modes import Discrete
# from rlbench.environment import Environment
# from rlbench.observation_config import ObservationConfig, CameraConfig
# from rlbench.tasks import ReachTarget
# from rlbench.tasks.taskboard_robothon import TaskboardRobothon


# class ImitationLearning(object):

#     def predict_action(self, batch):
#         return np.random.uniform(size=(len(batch), 7))

#     def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
#         return 1


# # To use 'saved' demos, set the path below, and set live_demos=False
# live_demos = False
# DATASET = '' if live_demos else '/media/sun/Expansion/Robot_learning'
# cam_config = CameraConfig(mask=False)
# class ImitationLearning(object):

#     def predict_action(self, batch):
#         return np.random.uniform(size=(len(batch), 7))

#     def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
#         return 1


# # To use 'saved' demos, set the path below, and set live_demos=False
# live_demos = False
# DATASET = '' if live_demos else '/media/sun/Expansion/Robot_learning'
# cam_config = CameraConfig(mask=False)
# obs_config = ObservationConfig(left_shoulder_camera = cam_config,
#                                right_shoulder_camera=cam_config,
#                                overhead_camera = cam_config,
#                                wrist_camera=cam_config,
#                                front_camera=cam_config)
# #obs_config.set_all(True)

# env = Environment(
#     action_mode=MoveArmThenGripper(
#         arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
#         dataset_root=DATASET,
#     obs_config=obs_config,#ObservationConfig(),
#     headless=False)
# env.launch()

# task = env.get_task(TaskboardRobothon)

# il = ImitationLearning()

# demos = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
# demos = np.array(demos).flatten()

# # An example of using the demos to 'train' using behaviour cloning loss.
# for i in range(100):
#     print("'training' iteration %d" % i)
#     batch = np.random.choice(demos, replace=False)
#     # print(batch.wrist_rgb)
#     batch_images = [obs for obs in batch.wrist_rgb]
#     predicted_actions = il.predict_action(batch_images)
#     ground_truth_actions = [obs for obs in batch.gripper_pose]
#     loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)

# print('Done')
# env.shutdown()
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
# obs_config = ObservationConfig(left_shoulder_camera = cam_config,
#                                right_shoulder_camera=cam_config,
#                                overhead_camera = cam_config,
#                                wrist_camera=cam_config,
#                                front_camera=cam_config)
# #obs_config.set_all(True)

# env = Environment(
#     action_mode=MoveArmThenGripper(
#         arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
#         dataset_root=DATASET,
#     obs_config=obs_config,#ObservationConfig(),
#     headless=False)
# env.launch()

# task = env.get_task(TaskboardRobothon)

# il = ImitationLearning()

# demos = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
# demos = np.array(demos).flatten()

# # An example of using the demos to 'train' using behaviour cloning loss.
# for i in range(100):
#     print("'training' iteration %d" % i)
#     batch = np.random.choice(demos, replace=False)
#     # print(batch.wrist_rgb)
#     batch_images = [obs for obs in batch.wrist_rgb]
#     predicted_actions = il.predict_action(batch_images)
#     ground_truth_actions = [obs for obs in batch.gripper_pose]
#     loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)

# print('Done')
# env.shutdown()
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
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
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
    demos = task.get_demos(1, live_demos=live_demos)
    # print(demos[0]._observations)
    # demos = np.array(demos).flatten()
    
    # Create dataset
    dataset = RLBenchDemoDataset(demos)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #len(train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize TensorBoard Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir='logs',
        name='rlbench_imitation_learning',
        version=None
    )
    
    # Initialize model
    model = ImitationLearningCNN(input_channels=3) #, pose_dim=7
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    # Initialize Trainer
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
            )#,
            #early_stop_callback
        ]
    )
    
    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    # Save the final model
    torch.save(model.state_dict(), 'rlbench_imitation_learning.pth')
    
    # Shutdown environment
    env.shutdown()
    
    print("Training complete!")

if __name__ == "__main__":
    main()