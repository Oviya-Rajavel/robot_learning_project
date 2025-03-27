import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import math
import os
import matplotlib.pyplot as plt
import io
import numpy as np
import matplotlib.image as mpimg

import torch
import numpy as np
import matplotlib.pyplot as plt
import io

def plot_trajectory(poses, predicted_poses, target_poses, prediction_horizon=1):
    """
    Create a 3D trajectory plot for TensorBoard visualization
    
    Args:
        images (torch.Tensor): Input image sequence
        poses (torch.Tensor): Input pose sequence
        predicted_poses (torch.Tensor): Predicted next pose
        target_poses (torch.Tensor): Ground truth next pose
        prediction_horizon (int): Number of timesteps to plot
    
    Returns:
        matplotlib figure with 3D trajectory visualization
    """
    # Ensure tensors are on CPU and converted to numpy
    poses = poses.cpu().numpy()
    predicted_poses = predicted_poses.cpu().numpy()
    target_poses = target_poses.cpu().numpy()
    
    # Squeeze and extract first batch if needed
    if predicted_poses.ndim > 2:
        predicted_poses = predicted_poses[0]
    if target_poses.ndim > 2:
        target_poses = target_poses[0]
    
    # Extract x, y, z coordinates (first 3 dimensions of pose)
    trajectory = poses[:, :, :3]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full trajectory
    ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 'b-o', label='Input Trajectory')
    
    # Highlight prediction horizon
    #horizon_x = trajectory[0, :prediction_horizon, 0]
    #horizon_y = trajectory[0, :prediction_horizon, 1]
    #horizon_z = trajectory[0, :prediction_horizon, 2]
    #ax.plot(horizon_x, horizon_y, horizon_z, 'g-', linewidth=3, label='Prediction Horizon')
    
    # Mark last known point
    last_point = trajectory[0, prediction_horizon-1]
    #ax.scatter(last_point[0], last_point[1], last_point[2], color='red', s=100, marker='x', label='Last Known Point')
    
    # Safely handle point extraction
    predicted_point = predicted_poses[:3] if predicted_poses.ndim == 1 else predicted_poses[0, :3]
    target_point = target_poses[:3] if target_poses.ndim == 1 else target_poses[0, :3]
    
    # Plot predicted and target points with connecting lines
    # Connect last known point to predicted and target points
    ax.plot([last_point[0], predicted_point[0]], 
            [last_point[1], predicted_point[1]], 
            [last_point[2], predicted_point[2]], 
            color='purple', linestyle='--', label='Predicted Path')
    
    ax.plot([last_point[0], target_point[0]], 
            [last_point[1], target_point[1]], 
            [last_point[2], target_point[2]], 
            color='orange', linestyle=':', label='Target Path')
    
    # Plot points with different markers and colors
    ax.scatter(predicted_point[0], predicted_point[1], predicted_point[2], 
               color='purple', marker='s', s=100, label='Predicted Next Pose')
    ax.scatter(target_point[0], target_point[1], target_point[2], 
               color='orange', marker='*', s=100, label='Target Next Pose')
    
    ax.set_title('3D Trajectory Prediction')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    
    # Save plot to buffer
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #plt.close()
    
    #return buf
    return fig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it's not a parameter but moves with the model
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Select the first seq_len rows of the positional encoding
        return self.pe[:x.size(1), :]

class PoseTransformerLightning(pl.LightningModule):
    def __init__(
        self, 
        image_feature_dim=512,  # ResNet feature dimension
        pose_dim=7,  # Assuming 7D pose (e.g., x,y,z + quaternion)
        d_model=256,  # Transformer model dimension
        nhead=8,      # Number of attention heads
        num_encoder_layers=2,
        dropout=0.1,
        max_seq_len=20,
        input_channels=3
    ):
        super().__init__()
        
       # Image Feature Extraction Layers
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
        
        # Image Feature Projection
        self.image_feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, d_model),
            nn.ReLU()
        )
        
        # Pose Embedding Layer
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Pose Prediction Head
        self.pose_prediction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pose_dim)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters()
        self.test_data = {
            "poses": [],
            "predicted_poses": [],
            "target_poses" : [],
            "loss": []
        }
    
    def forward(self, images, poses):
        """
        Forward pass for sequence-to-point pose prediction
        
        Args:
            images (torch.Tensor): Sequence of images 
                Shape: [batch_size, sequence_length, channels, height, width]
            poses (torch.Tensor): Sequence of poses 
                Shape: [batch_size, sequence_length, pose_dim]
        
        Returns:
            torch.Tensor: Predicted next pose
                Shape: [batch_size, pose_dim]
        """
        batch_size, seq_len, c, h, w = images.shape
        
        # Extract image features and project to d_model dimension
        image_features = []
        for t in range(seq_len):
            img_feat = self.image_features(images[:, t, :, :, :])
            projected_img_feat = self.image_feature_projector(img_feat)
            image_features.append(projected_img_feat)
        image_features = torch.stack(image_features, dim=1)  # [batch_size, seq_len, d_model]
        
        # Embed poses
        pose_features = self.pose_embedding(poses)
        
        # Combine image and pose features
        combined_features = image_features + pose_features
        
        # Add positional encoding
        pos_encoded_features = combined_features + self.positional_encoding(combined_features)
        
        # Apply dropout
        pos_encoded_features = self.dropout(pos_encoded_features)
        
        # Reshape for transformer (sequence, batch, features)
        pos_encoded_features = pos_encoded_features.permute(1, 0, 2)
        
        # Pass through transformer encoder
        encoded_features = self.transformer_encoder(pos_encoded_features)
        
        # Take the last timestep's features for prediction
        last_timestep_features = encoded_features[-1]
        
        # Predict next pose
        predicted_pose = self.pose_prediction_head(last_timestep_features)
        
        return predicted_pose

    
    # def training_step(self, batch, batch_idx):
    #     """
    #     Training step for PyTorch Lightning
        
    #     Args:
    #         batch (tuple): Batch of data 
    #             (images, poses, target_poses)
    #         batch_idx (int): Batch index
        
    #     Returns:
    #         torch.Tensor: Computed loss
    #     """
    #     images, poses, target_poses = batch
        
    #     # Predict next pose
    #     predicted_poses = self(images, poses)
        
    #     # Compute loss
    #     loss = self.loss_fn(predicted_poses, target_poses)
        
    #     # Log metrics
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
    #     return loss

    def quaternion_loss(self, q_pred, q_true):
        """
        Compute the loss between predicted and true quaternions using dot product.
        """
        dot_product = torch.sum(q_pred * q_true, dim=-1)  # Compute dot product
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp values to avoid NaN due to precision
        angle = torch.acos(dot_product)  # Compute angular distance
        return torch.mean(angle ** 2)  # Return squared angular distance as the loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning
        
        Args:
            batch (tuple): Batch of data Train
                (images, poses, target_poses)
            batch_idx (int): Batch index
        
        Returns:
            torch.Tensor: Computed loss
        """
        images, poses, target_poses = batch
        
        # Predict next pose
        predicted_poses = self(images, poses)  # Shape: (batch_size, T, 7)
        
        # Separate position and quaternion components (assuming 7 values per pose)
        predicted_poses = self(images, poses)  # Shape: [batch_size, 7]
        
        # Separate position (x, y, z) and quaternion (qx, qy, qz, qw)
        predicted_positions = predicted_poses[:, :3]*100  # First 3 values: [x, y, z]
        predicted_quaternions = predicted_poses[:, 3:]  # Last 4 values: [qx, qy, qz, qw]
        
        target_positions = target_poses[:, :3]*100  # Same: [x, y, z]
        target_quaternions = target_poses[:, 3:]  # Same: [qx, qy, qz, qw]
        
        # Compute position loss (Euclidean distance) for the batch
        position_loss = self.loss_fn(predicted_positions, target_positions)

        position_loss=position_loss*100
        
        # Compute quaternion loss (angular difference) for the batch
        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)
        
        # Total loss: sum of position and orientation loss
        total_loss = position_loss + quaternion_loss

        total_loss=total_loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with trajectory visualization
        
        Args:
            batch (tuple): Batch of data 
                (images, poses, target_poses)
            batch_idx (int): Batch index
        
        Returns:
            torch.Tensor: Computed loss
        """
        images, poses, target_poses = batch
        
        # Predict next pose
        predicted_poses = self(images, poses)
        
        # Compute loss
        loss = self.loss_fn(predicted_poses, target_poses)

        predicted_positions = predicted_poses[:, :3]*100  # First 3 values: [x, y, z]
        predicted_quaternions = predicted_poses[:, 3:]  # Last 4 values: [qx, qy, qz, qw]
        
        target_positions = target_poses[:, :3]*100  # Same: [x, y, z]
        target_quaternions = target_poses[:, 3:]  # Same: [qx, qy, qz, qw]
        
        # Compute position loss (Euclidean distance) for the batch
        position_loss = self.loss_fn(predicted_positions, target_positions)

        # Compute quaternion loss (angular difference) for the batch
        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)
        
        # Total loss: sum of position and orientation loss
        total_val_loss = position_loss + quaternion_loss
        
        # Log metrics
        self.log('val_loss', total_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Visualization (do this only for the first few batches to avoid overhead)
        if batch_idx < 3:  # Adjust number of visualizations as needed
            # Assuming prediction_horizon is the sequence length - 1
            prediction_horizon = images.shape[1]
            
            # Create trajectory plot
            fig = plot_trajectory(
                poses, 
                predicted_poses.unsqueeze(0), 
                target_poses.unsqueeze(0), 
                prediction_horizon
            )
            
            # Convert buffer to numpy array
            #trajectory_plot = plt.imread(trajectory_plot_buf)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            
            img = mpimg.imread(buf)

            # Log image to TensorBoard
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_image(
                f'trajectory_prediction/batch_{batch_idx}', 
                img.transpose(2, 0, 1), 
                global_step=self.global_step
            )
        
        return total_val_loss
    
    def test_step(self, batch, batc_idx):
        images, poses, target_poses = batch
        predicted_poses = self(images, poses)
        #loss = self.loss_fn(predicted_poses, target_poses)

        predicted_positions = predicted_poses[:, :3]*100  # First 3 values: [x, y, z]
        predicted_quaternions = predicted_poses[:, 3:]  # Last 4 values: [qx, qy, qz, qw]
        
        target_positions = target_poses[:, :3]*100  # Same: [x, y, z]
        target_quaternions = target_poses[:, 3:]  # Same: [qx, qy, qz, qw]
        
        # Compute position loss (Euclidean distance) for the batch
        position_loss = self.loss_fn(predicted_positions, target_positions)

        # Compute quaternion loss (angular difference) for the batch
        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)
        
        # Total loss: sum of position and orientation loss
        total_test_loss = position_loss + quaternion_loss
        
        self.test_data["poses"].append(poses.detach().cpu())
        self.test_data["predicted_poses"].append(predicted_poses.detach().cpu())
        self.test_data["target_poses"].append(target_poses.detach().cpu())
        self.test_data["loss"].append(total_test_loss)
        return total_test_loss
    
    def on_test_epoch_end(self):
        all_true_poses = torch.cat(self.test_data["poses"])
        all_pred_samples = torch.cat(self.test_data["predicted_poses"])
        all_targets = torch.cat(self.test_data["target_poses"])
        avg_test_loss = torch.mean(torch.tensor(self.test_data["loss"]))
        self.log('test_loss', avg_test_loss, prog_bar=True)

        for i in range(len(all_true_poses)):
            fig = plot_trajectory(
                all_true_poses[i:i+1], 
                all_pred_samples[i:i+1], 
                all_targets[i:i+1]
            )
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

    def configure_optimizers(self):
        """
        Configure optimizer for training
        
        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3,  # Learning rate
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# def train_pose_transformer(demos, 
#                             max_epochs=50, 
#                             batch_size=32, 
#                             log_dir='./lightning_logs1'):
#     """
#     Train the Pose Transformer using PyTorch Lightning
    
#     Args:
#         demos (list): List of RLBench demo objects
#         max_epochs (int): Maximum number of training epochs
#         batch_size (int): Batch size for training
#         log_dir (str): Directory for TensorBoard logs
#     """
#     # Create dataloader (using the sequence prediction dataloader)
#     train_loader, val_loader, test_loader = create_sequence_prediction_dataloaders(
#         demos, 
#         window_size=10,        # Number of timesteps in input sequence
#         prediction_offset=1,   # Predict 1 timestep ahead
#         batch_size=batch_size  # Batch size
#     )

#     # Initialize model
#     model = PoseTransformerLightning()
    
#     # TensorBoard Logger
#     logger = pl.loggers.TensorBoardLogger(
#         save_dir=log_dir, 
#         name='pose_transformer'
#     )
    
#     # Checkpointing
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         monitor='val_loss',
#         dirpath=os.path.join(log_dir, 'checkpoints'),
#         filename='pose_transformer-{epoch:02d}-{val_loss:.2f}',
#         save_top_k=3,
#         mode='min'
#     )
    
#     # Early stopping
#     early_stop_callback = pl.callbacks.EarlyStopping(
#         monitor='val_loss',
#         min_delta=0.00,
#         patience=10,
#         verbose=True,
#         mode='min'
#     )
    
#     # Trainer
#     trainer = pl.Trainer(
#         max_epochs=max_epochs,
#         logger=logger,
#         callbacks=[checkpoint_callback, early_stop_callback],
#         accelerator='auto',
#         devices=1 if torch.cuda.is_available() else None,
#         check_val_every_n_epoch=1
#     )
    
#     # Train the model
#     trainer.fit(
#         model, 
#         train_dataloaders=train_loader, 
#         val_dataloaders=val_loader
#     )
    
#     return model