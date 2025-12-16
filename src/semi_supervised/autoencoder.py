import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class TemporalConvVAE(nn.Module):
    def __init__(self, n_features, latent_dim=128, hidden_channels=[256, 256, 256, 256]):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        encoder_layers = []
        in_ch = n_features
        dilations = [1, 2, 4, 8]

        for out_ch, dilation in zip(hidden_channels, dilations):
            encoder_layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size=3, dilation=dilation))
            in_ch = out_ch

        self.encoder = nn.Sequential(*encoder_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels[-1], latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_channels[-1])

        decoder_layers = []
        hidden_channels_rev = hidden_channels[::-1]

        for i in range(len(hidden_channels_rev) - 1):
            decoder_layers.append(
                nn.ConvTranspose1d(
                    hidden_channels_rev[i],
                    hidden_channels_rev[i + 1],
                    kernel_size=3,
                    padding=1
                )
            )
            decoder_layers.append(nn.BatchNorm1d(hidden_channels_rev[i + 1]))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Dropout(0.1))

        self.decoder = nn.Sequential(*decoder_layers)

        self.output_conv = nn.Conv1d(hidden_channels[0], n_features, kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.global_pool(h).squeeze(-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h = self.fc_decode(z)
        h = h.unsqueeze(-1).repeat(1, 1, seq_len)
        h = self.decoder(h)
        return self.output_conv(h)

    def forward(self, x):
        seq_len = x.size(2)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar

    def get_latent(self, x):
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


class PoseDataset(Dataset):
    def __init__(self, features_list, seq_len=90):
        self.features_list = features_list
        self.seq_len = seq_len
        self.max_features = max(f.shape[1] for f in features_list)

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        features = self.features_list[idx]

        if len(features) < self.seq_len:
            pad_len = self.seq_len - len(features)
            features = np.vstack([features, np.zeros((pad_len, features.shape[1]))])
        elif len(features) > self.seq_len:
            start = np.random.randint(0, len(features) - self.seq_len + 1)
            features = features[start:start + self.seq_len]

        if features.shape[1] < self.max_features:
            pad_features = self.max_features - features.shape[1]
            features = np.hstack([features, np.zeros((features.shape[0], pad_features))])

        features = torch.FloatTensor(features).T
        return features


def vae_loss_function(recon_x, x, mu, logvar, beta=0.5):
    batch_size = x.size(0)

    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kld_loss


class AutoencoderTrainer:
    def __init__(self, config, n_features, device='cuda'):
        self.config = config
        self.device = device
        self.n_features = n_features

        self.model = TemporalConvVAE(
            n_features=n_features,
            latent_dim=config.AUTOENCODER_LATENT_DIM,
            hidden_channels=config.AUTOENCODER_HIDDEN_CHANNELS
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.AUTOENCODER_LR,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(batch)
            loss = vae_loss_function(recon, batch, mu, logvar, beta=self.config.AUTOENCODER_BETA)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / n_batches

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                recon, mu, logvar = self.model(batch)
                loss = vae_loss_function(recon, batch, mu, logvar, beta=self.config.AUTOENCODER_BETA)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(self, train_features, val_features=None):
        print(f"\nTraining Autoencoder on {len(train_features)} videos")

        train_dataset = PoseDataset(train_features, seq_len=self.config.AUTOENCODER_SEQ_LEN)
        actual_n_features = train_dataset.max_features

        if actual_n_features != self.n_features:
            print(f"Adjusting model: expected {self.n_features}, got {actual_n_features} features")
            self.n_features = actual_n_features
            self.model = TemporalConvVAE(
                n_features=actual_n_features,
                latent_dim=self.config.AUTOENCODER_LATENT_DIM,
                hidden_channels=self.config.AUTOENCODER_HIDDEN_CHANNELS
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.AUTOENCODER_LR,
                weight_decay=1e-5
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )

        print(f"Feature dimension: {self.n_features}")
        print(f"Latent dimension: {self.config.AUTOENCODER_LATENT_DIM}")
        print(f"Device: {self.device}")

        num_workers = 0 if len(train_dataset) < 100 else 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.AUTOENCODER_BATCH_SIZE, len(train_dataset)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )

        val_loader = None
        if val_features is not None:
            val_dataset = PoseDataset(val_features, seq_len=self.config.AUTOENCODER_SEQ_LEN)
            val_num_workers = 0 if len(val_dataset) < 100 else 4
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.config.AUTOENCODER_BATCH_SIZE, len(val_dataset)),
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=True if val_num_workers > 0 else False
            )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.config.AUTOENCODER_EPOCHS + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{self.config.AUTOENCODER_EPOCHS} - Train Loss: {train_loss:.6f}")

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.6f}")

                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best')
                else:
                    patience_counter += 1

                if patience_counter >= self.config.AUTOENCODER_PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.scheduler.step(train_loss)

        self.save_model('final')
        print("Autoencoder training complete!")

    def extract_embeddings(self, features_list):
        self.model.eval()
        embeddings = []

        dataset = PoseDataset(features_list, seq_len=self.config.AUTOENCODER_SEQ_LEN)
        num_workers = 0 if len(dataset) < 100 else 4
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.config.AUTOENCODER_BATCH_SIZE, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                batch = batch.to(self.device)
                mu = self.model.get_latent(batch)
                embeddings.append(mu.cpu().numpy())

        return np.vstack(embeddings)

    def save_model(self, name='model'):
        path = self.config.MODEL_DIR / f'autoencoder_{name}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_features': self.n_features,
            'config': {
                'latent_dim': self.config.AUTOENCODER_LATENT_DIM,
                'hidden_channels': self.config.AUTOENCODER_HIDDEN_CHANNELS
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, name='best'):
        path = self.config.MODEL_DIR / f'autoencoder_{name}.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
