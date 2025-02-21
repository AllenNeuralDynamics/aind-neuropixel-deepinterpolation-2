import torch
import torch.nn as nn
import numpy as np
import zarr
from torch.utils.data import Dataset, DataLoader
import os
import numcodecs

# Network Library
class InterpolationNetworkLibrary:
    @staticmethod
    def sparse_unet_2d(in_channels, out_channels=1, latent_size=1024, grid_height=4, grid_width=192):
        """
        in_channels: 2*N_FRAMES
        latent_size: Intermediate size for sparse embedding
        grid_height, grid_width: Fixed grid dimensions (e.g., 4x192)
        """
        class SparseUNet2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_proj = nn.Linear(384, latent_size)
                self.pos_attention = nn.Sequential(
                    nn.Linear(latent_size, latent_size),
                    nn.ReLU(),
                    nn.Linear(latent_size, grid_height * grid_width),  # 4 * 192 = 768 positions
                    nn.Softmax(dim=-1)
                )
                self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), padding=1)
                self.pool1 = nn.MaxPool2d((2, 2))
                self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
                self.pool2 = nn.MaxPool2d((2, 2))
                self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
                self.pool3 = nn.MaxPool2d((2, 2))
                self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)
                # No pool4 since grid_height=4 can't be reduced further
                self.conv5 = nn.Conv2d(512, 1024, (3, 3), padding=1)
                self.up1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
                self.conv7 = nn.Conv2d(1024 + 512, 512, (3, 3), padding=1)
                self.up2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
                self.conv8 = nn.Conv2d(512 + 256, 256, (3, 3), padding=1)
                self.up3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
                self.conv9 = nn.Conv2d(256 + 128, 128, (3, 3), padding=1)
                # No up4 needed since grid_height=4 is already reached
                self.conv10 = nn.Conv2d(128 + 64, 64, (3, 3), padding=1)
                self.decoded = nn.Conv2d(64, out_channels, (1, 1), padding=0)
                self.final_dense = nn.Linear(grid_height * grid_width, 384)  # 768 -> 384
                self.relu = nn.ReLU()
                self.grid_height = grid_height
                self.grid_width = grid_width
                
            def forward(self, x):
                batch_size = x.size(0)
                latent = self.relu(self.latent_proj(x))
                pos_scores = self.pos_attention(latent.mean(dim=1))  # (batch, grid_height*grid_width)
                x_2d = torch.zeros(batch_size, in_channels, self.grid_height, self.grid_width, 
                                 device=x.device)
                for i in range(batch_size):
                    grid_flat = torch.einsum('cl,g->g', latent[i], pos_scores[i])
                    x_2d[i] = grid_flat.view(1, self.grid_height, self.grid_width).expand(in_channels, -1, -1)
                
                conv1 = self.relu(self.conv1(x_2d))  # (batch, 64, 4, 192)
                pool1 = self.pool1(conv1)  # (batch, 64, 2, 96)
                conv2 = self.relu(self.conv2(pool1))  # (batch, 128, 2, 96)
                pool2 = self.pool2(conv2)  # (batch, 128, 1, 48)
                conv3 = self.relu(self.conv3(pool2))  # (batch, 256, 1, 48)
                pool3 = self.pool3(conv3)  # (batch, 256, 1, 24) - height can't go below 1
                conv4 = self.relu(self.conv4(pool3))  # (batch, 512, 1, 24)
                conv5 = self.relu(self.conv5(conv4))  # (batch, 1024, 1, 24)
                up1 = self.up1(conv5)  # (batch, 1024, 2, 48)
                conc_up_1 = torch.cat([up1, conv3], dim=1)  # (batch, 1280, 2, 48)
                conv7 = self.relu(self.conv7(conc_up_1))  # (batch, 512, 2, 48)
                up2 = self.up2(conv7)  # (batch, 512, 4, 96)
                # Adjust conv2 size to match up2 for concatenation
                conv2_adjusted = nn.functional.interpolate(conv2, size=(4, 96), mode='bilinear', align_corners=False)
                conc_up_2 = torch.cat([up2, conv2_adjusted], dim=1)  # (batch, 640, 4, 96)
                conv8 = self.relu(self.conv8(conc_up_2))  # (batch, 256, 4, 96)
                up3 = self.up3(conv8)  # (batch, 256, 8, 192)
                # Adjust conv1 size to match up3 for concatenation
                conv1_adjusted = nn.functional.interpolate(conv1, size=(8, 192), mode='bilinear', align_corners=False)
                conc_up_3 = torch.cat([up3, conv1_adjusted], dim=1)  # (batch, 320, 8, 192)
                conv9 = self.relu(self.conv9(conc_up_3))  # (batch, 128, 8, 192)
                conv10 = self.relu(self.conv10(conv9))  # (batch, 64, 8, 192)
                decoded = self.decoded(conv10)  # (batch, 1, 8, 192)
                decoded = decoded.view(batch_size, 1, -1)  # (batch, 1, 4*192)
                out = self.final_dense(decoded)  # (batch, 1, 384)
                return out.squeeze(1), pos_scores
        
        return SparseUNet2D()

# Function to create fake zarr data
def create_fake_zarr_data(filepath, num_frames=1000, num_channels=384):
    clean_data = np.zeros((num_frames, num_channels), dtype=np.float32)
    for t in range(num_frames):
        clean_data[t] = np.sin(t * 0.1 + np.arange(num_channels) * 0.01)
    
    noisy_data = clean_data + np.random.normal(0, 0.1, clean_data.shape)
    
    zarr_group = zarr.open(filepath, mode='w')
    zarr_store = zarr_group.create_dataset('traces_seg0', shape=noisy_data.shape, 
                                          chunks=(1000, num_channels), dtype='f4')
    zarr_store[:] = noisy_data
    return zarr_store

# Custom Dataset for zarr files
class ZarrInterpolationDataset(Dataset):
    def __init__(self, zarr_path, array_name='traces_seg0', n_frames=2, chunk_size=1000):
        zarr_group = zarr.open(zarr_path, mode='r')
        try:
            self.zarr_data = zarr_group[array_name]
        except KeyError:
            print(f"Available arrays in zarr group: {list(zarr_group.array_keys())}")
            raise KeyError(f"Array '{array_name}' not found in zarr file {zarr_path}")
        except ValueError as e:
            if 'codec not available' in str(e):
                print(f"Codec error: {e}. Please install required codec (e.g., 'pip install numcodecs[wavpack]')")
            raise
        
        self.n_frames = n_frames
        self.chunk_size = chunk_size
        self.total_frames = self.zarr_data.shape[0]
        self.valid_indices = np.arange(n_frames, self.total_frames - n_frames)
        np.random.shuffle(self.valid_indices)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        input_frames = []
        for j in range(-self.n_frames, self.n_frames + 1):
            if j != 0:
                frame = self.zarr_data[center_idx + j]
                input_frames.append(frame)
        
        target_frame = self.zarr_data[center_idx]
        return torch.FloatTensor(np.stack(input_frames)), torch.FloatTensor(target_frame)

# Training function with grid visualization
def train_model(model, dataloader, num_epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs, pos_scores = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                threshold = 0.01
                active_positions = (pos_scores[0] > threshold).sum().item()
                max_score = pos_scores[0].max().item()
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                print(f"  Active grid positions (> {threshold}): {active_positions}/{model.grid_height * model.grid_width}")
                print(f"  Max position score: {max_score:.4f}")
                
                # Visualize the grid
                grid = pos_scores[0].view(model.grid_height, model.grid_width).cpu().numpy()
                print(f"  Grid ({model.grid_height}x{model.grid_width}, rounded to 3 decimals):")
                print(np.round(grid, 3))
        
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
            running_loss = 0.0

# Inference function
def denoise_frame(model, zarr_data, frame_idx, n_frames):
    model.eval()
    with torch.no_grad():
        input_frames = []
        for j in range(-n_frames, n_frames + 1):
            if j != 0:
                frame = zarr_data[frame_idx + j]
                input_frames.append(frame)
        input_stack = torch.FloatTensor(np.stack(input_frames)).unsqueeze(0).to(device)
        denoised, _ = model(input_stack)
        return denoised.squeeze().cpu().numpy()

# Main execution
if __name__ == "__main__":
    try:
        numcodecs.wavpack
    except AttributeError:
        print("Warning: 'wavpack' codec not available. Install with 'pip install numcodecs[wavpack]' if needed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N_FRAMES = 4
    BATCH_SIZE = 50
    ZARR_PATH = "/data/ecephys_660948_2023-05-01_20-02-08/ecephys_compressed/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA.zarr"
    ARRAY_NAME = 'traces_seg0'
    
    if not os.path.exists(ZARR_PATH):
        print("Creating fake zarr data...")
        create_fake_zarr_data(ZARR_PATH)
    else:
        print("Using existing zarr data...")
    
    zarr_group = zarr.open(ZARR_PATH, mode='r')
    print("Zarr group contents:", list(zarr_group.array_keys()))
    try:
        zarr_data = zarr_group[ARRAY_NAME]
        print(f"Data array shape: {zarr_data.shape}")
    except ValueError as e:
        print(f"Error accessing array: {e}")
        print("Likely a codec issue. Please ensure 'wavpack' is installed if required.")
        raise

    dataset = ZarrInterpolationDataset(ZARR_PATH, array_name=ARRAY_NAME, n_frames=N_FRAMES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    net_library = InterpolationNetworkLibrary()
    model = net_library.sparse_unet_2d(in_channels=2*N_FRAMES, latent_size=1024, 
                                     grid_height=4, grid_width=192).to(device)
    
    train_model(model, dataloader)
    
    zarr_data = zarr.open(ZARR_PATH, mode='r')[ARRAY_NAME]
    test_frame_idx = 50
    denoised_frame = denoise_frame(model, zarr_data, test_frame_idx, N_FRAMES)
    
    print(f"Denoised frame shape: {denoised_frame.shape}")