import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length demo batches
    
    Args:
        batch (list): List of (images, poses) tuples
    
    Returns:
        tuple: Batched images and poses
    """
    # Separate images and poses
    images = [item[0] for item in batch]
    poses = [item[1] for item in batch]
    
    # Find the maximum sequence length and maximum channels
    max_seq_len = max(img.shape[0] for img in images)
    max_channels = max(img.shape[1] for img in images)
    
    # Pad images and poses to the same length and channel count
    padded_images = []
    padded_poses = []
    
    for img, pose in zip(images, poses):
        seq_len, channels, height, width = img.shape
        
        # Pad sequence length
        pad_length = max_seq_len - seq_len
        padded_img = torch.nn.functional.pad(
            img, 
            (0, 0, 0, 0, 0, 0, 0, pad_length),  # Pad only the sequence dimension
            mode='constant', 
            value=0
        )
        
        # Pad channels if needed (repeat channels to match max_channels)
        if channels < max_channels:
            repeat_factor = max_channels // channels  # How many times to repeat
            padded_img = padded_img.repeat(1, repeat_factor, 1, 1)
            padded_img = padded_img[:, :max_channels, :, :]  # Trim extra channels
        
        # Pad poses
        pad_length = max_seq_len - pose.shape[0]
        padded_pose = torch.nn.functional.pad(
            pose, 
            (0, 0, 0, pad_length), 
            mode='constant', 
            value=0
        )
        
        padded_images.append(padded_img)
        padded_poses.append(padded_pose)
    
    # Stack padded images and poses
    batched_images = torch.stack(padded_images)
    batched_poses = torch.stack(padded_poses)
    
    return batched_images, batched_poses


class RLBenchDemoDatasetBatcher(Dataset):
    """
    A dataset batcher that creates batches of entire demos
    
    Args:
        demos (list): List of RLBench demo objects
    """
    def __init__(self, demos):
        """
        Initialize the batcher by preparing demos for batching
        
        Args:
            demos (list): List of RLBench demo objects
        """
        # Prepare demos with their images and gripper poses
        self.demos = []
        for demo in demos:
            # Extract images and gripper poses for this demo
            images = [obs.wrist_rgb for obs in demo._observations]
            gripper_poses = [obs.gripper_pose for obs in demo._observations]
            
            # Convert to torch tensors
            images_tensor = torch.FloatTensor(np.array(images) / 255.0).permute(0, 3, 1, 2)  # Normalize and change to NCHW
            gripper_poses_tensor = torch.FloatTensor(np.array(gripper_poses))
            
            # Store the demo as a tuple of tensors
            self.demos.append((images_tensor, gripper_poses_tensor))
    
    def __len__(self):
        """
        Returns the number of demos
        
        Returns:
            int: Number of demos
        """
        return len(self.demos)
    
    def __getitem__(self, idx):
        """
        Retrieve an entire demo (all images and gripper poses)
        
        Args:
            idx (int): Index of the demo
        
        Returns:
            tuple: (images_tensor, gripper_poses_tensor) for the entire demo
        """
        return self.demos[idx]

# def create_demo_batch_dataloader(demos, batch_size=1, shuffle=True, num_workers=0):
#     """
#     Create a DataLoader that batches entire demos
    
#     Args:
#         demos (list): List of RLBench demo objects
#         batch_size (int, optional): Number of demos per batch. Defaults to 1.
#         shuffle (bool, optional): Whether to shuffle the demos. Defaults to True.
#         num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
    
#     Returns:
#         DataLoader: A DataLoader that yields entire demos as batches
#     """
#     # Create the demo batcher dataset
#     #demo_dataset = RLBenchDemoDatasetBatcher(demos)
#     #
#     ## Create and return the DataLoader
#     #return DataLoader(
#     #    demo_dataset, 
#     #    batch_size=batch_size, 
#     #    shuffle=shuffle, 
#     #    num_workers=num_workers
#     #)



def create_dataloaders(demos, batch_size=1, num_workers=0, train_ratio=0.8, val_ratio=0.15, shuffle=True):
    """
    Split the dataset into train, validation, and test sets with corresponding DataLoaders.
    
    :param demos: Original dataset of demonstrations
    :param batch_size: Batch size for DataLoaders
    :param num_workers: Number of worker processes for DataLoader
    :param train_ratio: Proportion of data for training (default 70%)
    :param val_ratio: Proportion of data for validation (default 15%)
    :return: Tuple of (train_loader, val_loader, test_loader)
    """
    # Create the base dataset
    demo_dataset = RLBenchDemoDatasetBatcher(demos)
    
    # Calculate dataset sizes
    total_size = len(demo_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        demo_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

# Example usage
def main():
    # Assuming you have your task and demos set up
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
    demos = task.get_demos(3, live_demos=live_demos)
    
    # Create the demo batch dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        demos, 
        batch_size=1,  # Each batch is one full demo
        shuffle=False,  # Optional: set to True if you want to randomize demo order
    )
    
    # Iterate through the dataloader
    for batch_idx, (batch_images, batch_gripper_poses) in enumerate(train_dataloader):
        print(f"Demo Batch {batch_idx}:")
        print(f"Images shape: {batch_images.shape}")
        print(f"Gripper poses shape: {batch_gripper_poses.shape}")
        
        # Example processing
        # batch_images will be a tensor of all images from one demo
        # batch_gripper_poses will be a tensor of all gripper poses from one demo
        
        # You can process each demo separately
        for demo_idx in range(batch_images.shape[0]):
            demo_images = batch_images[demo_idx]
            demo_poses = batch_gripper_poses[demo_idx]
            
            # Process or analyze the entire demo
            print(f"  Demo {demo_idx} - Images: {demo_images.shape}, Poses: {demo_poses.shape}")
        

if __name__ == "__main__":
    main()