
import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F

def plot_trajectory(true_poses, predicted_poses, title='Gripper Trajectory'):
    """
    Create a matplotlib figure comparing true and predicted trajectories
    
    Args:
    - true_poses (torch.Tensor): Ground truth poses, shape [batch_size, 7]
    - predicted_poses (torch.Tensor): Predicted poses, shape [batch_size, 7]
    - title (str): Plot title
    
    Returns:
    - figure (matplotlib.figure.Figure)
    """
    # Convert to numpy for plotting
    true_poses = true_poses.cpu().numpy()
    predicted_poses = predicted_poses.cpu().numpy()
    
    # Create a figure with subplots for each of the 7 pose values
    # fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
    # fig.suptitle(title)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_poses[:, 0], true_poses[:, 1], true_poses[:, 2], marker="o", linestyle="-", color="b", label="Ground Truth")
    ax.plot(predicted_poses[:, 0], predicted_poses[:, 1], predicted_poses[:, 2], marker="o", linestyle="-", color="r", label="Predicted path")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Gripper Trajectory")
    ax.legend()
    # plt.show()
    return fig

class RLBenchDemoDataset(Dataset):
    def __init__(self, demos):
        """
        Dataset for RLBench demonstrations with sequential pose prediction
        
        Args:
        - demos: RLBench demonstrations
        """
        # Extract wrist camera RGB images and gripper poses
        self.images = []
        self.current_poses = []
        self.next_poses = []
        
        for demo in demos:
            for i in range(len(demo._observations) - 1):  # Stop at second to last observation
                # Current image
                current_image = demo._observations[i].wrist_rgb
                # Current pose
                current_pose = demo._observations[i].gripper_pose
                # Next pose
                next_pose = demo._observations[i+1].gripper_pose
                
                self.images.append(current_image)
                self.current_poses.append(current_pose)
                self.next_poses.append(next_pose)
        
        # Convert to torch tensors
        self.images = torch.FloatTensor(np.array(self.images) / 255.0).permute(0, 3, 1, 2)  # Normalize and change to NCHW
        self.current_poses = torch.FloatTensor(np.array(self.current_poses))
        self.next_poses = torch.FloatTensor(np.array(self.next_poses))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return (
            self.images[idx],  # Current image
            self.current_poses[idx],  # Current pose
            self.next_poses[idx]  # Next pose (target)
        )

class ImitationLearningCNN(pl.LightningModule):
    def __init__(self, input_channels=3, pose_input_dim=7, pose_output_dim=7, 
                 learning_rate=1e-4, lambda_pos=1.0, lambda_quat=1.0):
        """
        CNN for Imitation Learning in RLBench with next pose prediction
        
        Args:
        - input_channels: number of image channels
        - pose_input_dim: dimension of current gripper pose vector
        - pose_output_dim: dimension of next gripper pose vector
        - learning_rate: optimization learning rate
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()

        self.lambda_pos = lambda_pos
        self.lambda_quat = lambda_quat
        
        # Feature extraction layers for image
        self.image_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier layers combining image features and current pose
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16 + pose_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Next pose prediction layer
        self.pose_predictor = nn.Linear(128, pose_output_dim)
        
        # Loss function
        self.criterion = nn.MSELoss()

        self.test_data = {
            "poses": [],
            "predicted_poses": [],
            "target_poses" : [],
            "loss": []
        }
    
    def forward(self, image, current_pose):
        """
        Forward pass of the network
        
        Args:
        - image: input image tensor
        - current_pose: current gripper pose tensor
        
        Returns:
        - Predicted next gripper pose
        """
        # Extract image features
        image_features = self.image_features(image)
        
        # Flatten image features
        image_flat = torch.flatten(image_features, start_dim=1)
        
        # Concatenate image features with current pose
        combined_features = torch.cat([image_flat, current_pose], dim=1)
        
        # Process combined features
        processed_features = self.classifier(combined_features)
        
        # Predict next pose
        next_pose_prediction = self.pose_predictor(processed_features)
        
        return next_pose_prediction
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning
        
        Args:
        - batch: tuple of (images, current_poses, target_next_poses)
        - batch_idx: batch index
        
        Returns:
        - loss
        """
        images, current_poses, target_next_poses = batch
        
        # Predict next poses
        predicted_next_poses = self(images, current_poses)
        
        # Separate position and quaternion components
        pred_pos, pred_quat = predicted_next_poses[:, :3], predicted_next_poses[:, 3:]
        target_pos, target_quat = target_next_poses[:, :3], target_next_poses[:, 3:]
        
        # Position Loss (MSE Loss)
        pos_loss = F.mse_loss(pred_pos, target_pos)
        
        # Quaternion Loss (1 - |dot product| to measure similarity)
        quat_loss = 1 - torch.abs(torch.sum(pred_quat * target_quat, dim=1)).mean()
        
        # Total loss
        loss = self.lambda_pos * pos_loss + self.lambda_quat * quat_loss

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Optional: Log some additional metrics or visualizations
        if batch_idx % 10 == 0:
            # Log histogram of predicted poses
            self.logger.experiment.add_histogram('predicted_next_poses', predicted_next_poses, self.global_step)
            
            # Log a sample image (first image in the batch)
            self.logger.experiment.add_image('sample_input_image', images[0], self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning
        
        Args:
        - batch: tuple of (images, current_poses, target_next_poses)
        - batch_idx: batch index
        """
        images, current_poses, target_next_poses = batch
        
        # Predict next poses
        predicted_next_poses = self(images, current_poses)
        
        # Compute validation loss
        val_loss = self.criterion(predicted_next_poses, target_next_poses)
        
        # Log validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Optional: Log mean absolute error
        mae = torch.mean(torch.abs(predicted_next_poses - target_next_poses))
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Log trajectory visualization (only for the first batch per epoch)
        if batch_idx == 0:
            # Select first few samples from the batch (e.g., first 3)self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
            true_sample = target_next_poses[:3] #[x, y, z, qx, qy, qz, qw]
            pred_sample = predicted_next_poses[:3]
            
            # Create trajectory plot
            fig = plot_trajectory(true_sample, pred_sample, title='Predicted Next Poses')
            
            # Convert plot to tensorboard image
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to numpy array
            import matplotlib.image as mpimg
            img = mpimg.imread(buf)
            
            # Log to TensorBoard
            self.logger.experiment.add_image(
                'validation_next_trajectory', 
                img.transpose(2, 0, 1),  # Convert to channels-first format
                global_step=self.global_step
            )
            
            # Close the figure to free up memory
            plt.close(fig)
    
        return val_loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        
        Returns:
        - Dictionary with optimizer and learning rate scheduler
        """
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=1e-5
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def test_step(self, batch, batc_idx):
        images, current_poses, target_next_poses = batch
        
        # Predict next poses
        predicted_next_poses = self(images, current_poses)
        
        # Compute validation loss
        test_loss = self.criterion(predicted_next_poses, target_next_poses)
        
        self.test_data["poses"].append(current_poses.detach().cpu())
        self.test_data["target_poses"].append(target_next_poses.detach().cpu())
        self.test_data["predicted_poses"].append(predicted_next_poses.detach().cpu())
        self.test_data["loss"].append(test_loss)
        return test_loss

    
    def on_test_epoch_end(self):
        all_true_poses = torch.cat(self.test_data["poses"])
        all_pred_samples = torch.cat(self.test_data["predicted_poses"])
        all_target_samples = torch.cat(self.test_data["target_poses"])
        avg_test_loss = torch.mean(torch.tensor(self.test_data["loss"]))
        self.log('test_loss', avg_test_loss, prog_bar=True)

        for i in range(len(all_target_samples)):
            fig = plot_trajectory(all_target_samples[i:i+1], 
                all_pred_samples[i:i+1], title='Predicted Next Poses')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            img = mpimg.imread(buf)
            self.logger.experiment.add_image(
                f'test_trajectory_{i}', 
                img.transpose(2, 0, 1),
                global_step=i
            )

            plt.close(fig)
