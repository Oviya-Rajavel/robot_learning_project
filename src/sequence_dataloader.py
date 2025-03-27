import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class SequencePredictionDataset(Dataset):
    """
    A dataset that creates sliding windows for sequence prediction
    
    Args:
        demos (list): List of RLBench demo objects
        window_size (int): Size of the input sequence window
        prediction_offset (int): Timesteps ahead to predict
    """
    def __init__(self, demos, window_size=20, prediction_offset=1):
        """
        Initialize the sequence prediction dataset
        
        Args:
            demos (list): List of RLBench demo objects
            window_size (int): Number of timesteps in input sequence
            prediction_offset (int): How many timesteps ahead to predict
        """
        self.windows = []
        
        for demo in demos:
            # Extract images and gripper poses for this demo
            images = [obs.wrist_rgb for obs in demo._observations]
            gripper_poses = [obs.gripper_pose for obs in demo._observations]
            
            # Convert to torch tensors
            images_tensor = torch.FloatTensor(np.array(images) / 255.0).permute(0, 3, 1, 2)  # Normalize and change to NCHW
            gripper_poses_tensor = torch.FloatTensor(np.array(gripper_poses))
            
            # Create sliding windows
            for start in range(0, len(images_tensor) - window_size - prediction_offset + 1):
                # Input sequence window
                
                input_images = images_tensor[start:start+window_size] #0:20 #19
                input_poses = gripper_poses_tensor[start:start+window_size]
                
                # Target pose to predict (prediction_offset steps ahead)
                target_pose = gripper_poses_tensor[start+window_size+prediction_offset-1]

                # Store the window, input poses, and target pose
                self.windows.append((
                    input_images,   # Sequence of images
                    input_poses,    # Sequence of poses
                    target_pose     # Target pose to predict
                ))
    
    def __len__(self):
        """
        Returns the number of windows
        
        Returns:
            int: Number of windows
        """
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Retrieve a specific window
        
        Args:
            idx (int): Index of the window
        
        Returns:
            tuple: (images_tensor, poses_tensor, target_pose)
        """
        return self.windows[idx]

def custom_sequence_prediction_collate_fn(batch):
    """
    Custom collate function to handle sequence prediction batches
    
    Args:
        batch (list): List of (images, poses, target) tuples
    
    Returns:
        tuple: Batched images, poses, and target poses
    """
    # Separate images, poses, and targets
    images = [item[0] for item in batch]
    poses = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    # Stack batched images and poses
    batched_images = torch.stack(images)
    batched_poses = torch.stack(poses)
    batched_targets = torch.stack(targets)
    
    return batched_images, batched_poses, batched_targets

def create_sequence_prediction_dataloaders(
    demos, 
    window_size=20, 
    prediction_offset=1,
    batch_size=32, 
    num_workers=0, 
    train_ratio=0.8, 
    val_ratio=0.15
):
    """
    Create dataloaders for sequence prediction
    
    Args:
        demos (list): List of RLBench demonstrations
        window_size (int): Size of input sequence window
        prediction_offset (int): Timesteps ahead to predict
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloader
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create sequence prediction dataset
    sequence_dataset = SequencePredictionDataset(
        demos, 
        window_size=window_size, 
        prediction_offset=prediction_offset
    )
    
    # Calculate dataset sizes
    total_size = len(sequence_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        sequence_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    return train_loader, val_loader, test_loader

# Example usage in main function
# def main():
#     # Your existing RLBench setup code...
    
#     # Create sequence prediction dataloaders
#     train_loader, val_loader, test_loader = create_sequence_prediction_dataloaders(
#         demos, 
#         window_size=20,        # Number of timesteps in input sequence
#         prediction_offset=1,   # Predict 1 timestep ahead
#         batch_size=32          # Batch size
#     )
    
#     # Iterate through the dataloader
#     for batch_idx, (batch_images, batch_poses, batch_targets) in enumerate(train_loader):
#         print(f"Batch {batch_idx}:")
#         print(f"Images shape: {batch_images.shape}")
#         print(f"Input Poses shape: {batch_poses.shape}")
#         print(f"Target Poses shape: {batch_targets.shape}")
        
#         # Shapes explained:
#         # batch_images: [batch_size, window_size, channels, height, width]
#         # batch_poses: [batch_size, window_size, pose_dimensions]
#         # batch_targets: [batch_size, pose_dimensions]