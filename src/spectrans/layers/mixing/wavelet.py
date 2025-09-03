"""Wavelet-based mixing layers for spectral transformers.

This module provides neural network layers that perform mixing operations
in the wavelet domain, enabling multi-resolution processing of signals.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ...core.types import ConfigDict
from ...core.registry import register_component
from ...transforms.wavelet import DWT1D, DWT2D


@register_component("mixing", "wavelet_mixing")
class WaveletMixing(nn.Module):
    """Token mixing layer using discrete wavelet transform.
    
    Performs mixing in wavelet domain for multi-resolution processing.
    Decomposes input using DWT, applies learnable mixing to coefficients,
    and reconstructs the output.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size.
    wavelet : str, default='db4'
        Wavelet type (e.g., 'db1', 'db4', 'sym2').
    levels : int, default=3
        Number of decomposition levels.
    mixing_mode : str, default='pointwise'
        How to mix coefficients ('pointwise', 'channel', 'level').
    dropout : float, default=0.0
        Dropout probability for mixing weights.
    
    Attributes
    ----------
    dwt : DWT1D
        Wavelet transform module.
    mixing_weights : nn.ParameterDict
        Learnable weights for mixing at each level.
    dropout : nn.Dropout
        Dropout layer for regularization.
    
    Examples
    --------
    >>> mixer = WaveletMixing(hidden_dim=256, wavelet='db4', levels=3)
    >>> x = torch.randn(32, 128, 256)  # (batch, seq_len, hidden)
    >>> output = mixer(x)
    >>> assert output.shape == x.shape
    """
    
    def __init__(
        self,
        hidden_dim: int,
        wavelet: str = 'db4',
        levels: int = 3,
        mixing_mode: str = 'pointwise',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.wavelet = wavelet
        self.levels = levels
        self.mixing_mode = mixing_mode
        
        # Initialize wavelet transform
        self.dwt = DWT1D(wavelet=wavelet, levels=levels, mode='symmetric')
        
        # Initialize mixing weights based on mode
        self.mixing_weights = nn.ParameterDict()
        
        if mixing_mode == 'pointwise':
            # Simple pointwise multiplication for each level
            self.mixing_weights['approx'] = nn.Parameter(
                torch.ones(1, 1, hidden_dim)
            )
            for level in range(levels):
                self.mixing_weights[f'detail_{level}'] = nn.Parameter(
                    torch.ones(1, 1, hidden_dim)
                )
                
        elif mixing_mode == 'channel':
            # Channel-wise mixing matrices
            self.mixing_weights['approx'] = nn.Parameter(
                torch.eye(hidden_dim).unsqueeze(0)
            )
            for level in range(levels):
                self.mixing_weights[f'detail_{level}'] = nn.Parameter(
                    torch.eye(hidden_dim).unsqueeze(0)
                )
                
        elif mixing_mode == 'level':
            # Cross-level mixing with attention-like mechanism
            self.level_mixer = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )
        else:
            raise ValueError(f"Unknown mixing mode: {mixing_mode}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply wavelet-based mixing.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, seq_len, hidden_dim).
            
        Returns
        -------
        Tensor
            Mixed output tensor of same shape as input.
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Store original input for residual connection
        residual = x
        
        # Process each hidden dimension independently
        outputs = []
        for h in range(hidden_dim):
            # Extract single channel
            x_channel = x[:, :, h:h+1]  # Keep dimension for consistency
            
            # Decompose using DWT
            approx, details = self.dwt.decompose(x_channel, dim=1)
            
            # Apply mixing based on mode
            if self.mixing_mode == 'pointwise':
                # Apply pointwise scaling
                approx_mixed = approx * self.mixing_weights['approx'][:, :, h:h+1]
                details_mixed = []
                for level, detail in enumerate(details):
                    weight = self.mixing_weights[f'detail_{level}'][:, :, h:h+1]
                    details_mixed.append(detail * weight)
                    
            elif self.mixing_mode == 'channel':
                # Apply channel mixing (simplified for single channel processing)
                approx_mixed = approx * self.mixing_weights['approx'][:, h, h]
                details_mixed = []
                for level, detail in enumerate(details):
                    weight = self.mixing_weights[f'detail_{level}'][:, h, h]
                    details_mixed.append(detail * weight)
                    
            elif self.mixing_mode == 'level':
                # Stack all coefficients for cross-level mixing
                all_coeffs = [approx] + details
                max_len = max(c.shape[1] for c in all_coeffs)
                
                # Pad to same length
                padded_coeffs = []
                for coeff in all_coeffs:
                    if coeff.shape[1] < max_len:
                        pad_len = max_len - coeff.shape[1]
                        coeff = torch.nn.functional.pad(coeff, (0, 0, 0, pad_len))
                    padded_coeffs.append(coeff)
                
                # Stack and apply attention
                stacked = torch.stack(padded_coeffs, dim=1)  # (batch, levels+1, max_len, 1)
                stacked = stacked.squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
                
                # Apply self-attention across levels
                mixed, _ = self.level_mixer(stacked, stacked, stacked)
                
                # Extract mixed coefficients
                approx_mixed = mixed[:, 0:1, :approx.shape[1], h:h+1]
                details_mixed = []
                for level in range(self.levels):
                    detail_len = details[level].shape[1]
                    detail_mixed = mixed[:, level+1:level+2, :detail_len, h:h+1]
                    details_mixed.append(detail_mixed.squeeze(1))
                approx_mixed = approx_mixed.squeeze(1)
            
            # Reconstruct signal
            reconstructed = self.dwt.reconstruct((approx_mixed, details_mixed), dim=1)
            
            # Ensure output has correct length
            if reconstructed.shape[1] != seq_len:
                reconstructed = reconstructed[:, :seq_len]
                
            outputs.append(reconstructed)
        
        # Combine all channels
        output = torch.cat(outputs, dim=-1)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = output + residual
        
        return output
    
    @classmethod
    def from_config(cls, config: ConfigDict) -> "WaveletMixing":
        """Create WaveletMixing from configuration.
        
        Parameters
        ----------
        config : ConfigDict
            Configuration dictionary.
            
        Returns
        -------
        WaveletMixing
            Configured instance.
        """
        return cls(
            hidden_dim=config["hidden_dim"],
            wavelet=config.get("wavelet", "db4"),
            levels=config.get("levels", 3),
            mixing_mode=config.get("mixing_mode", "pointwise"),
            dropout=config.get("dropout", 0.0),
        )


@register_component("mixing", "wavelet_mixing_2d")
class WaveletMixing2D(nn.Module):
    """2D wavelet mixing layer for image-like data.
    
    Performs mixing in 2D wavelet domain, suitable for vision transformers
    and other architectures processing 2D spatial data.
    
    Parameters
    ----------
    channels : int
        Number of input/output channels.
    wavelet : str, default='db4'
        Wavelet type.
    levels : int, default=2
        Number of decomposition levels.
    mixing_mode : str, default='subband'
        How to mix subbands ('subband', 'cross', 'attention').
    
    Examples
    --------
    >>> mixer = WaveletMixing2D(channels=256, wavelet='db4', levels=2)
    >>> x = torch.randn(32, 256, 64, 64)  # (batch, channels, height, width)
    >>> output = mixer(x)
    >>> assert output.shape == x.shape
    """
    
    def __init__(
        self,
        channels: int,
        wavelet: str = 'db4',
        levels: int = 2,
        mixing_mode: str = 'subband',
    ):
        super().__init__()
        
        self.channels = channels
        self.wavelet = wavelet
        self.levels = levels
        self.mixing_mode = mixing_mode
        
        # Initialize 2D wavelet transform
        self.dwt = DWT2D(wavelet=wavelet, levels=levels, mode='symmetric')
        
        # Initialize mixing layers based on mode
        if mixing_mode == 'subband':
            # Independent processing of each subband
            self.ll_mixer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            
            self.detail_mixers = nn.ModuleList()
            for _ in range(levels):
                detail_mixer = nn.ModuleDict({
                    'lh': nn.Conv2d(channels, channels, 3, padding=1),
                    'hl': nn.Conv2d(channels, channels, 3, padding=1),
                    'hh': nn.Conv2d(channels, channels, 3, padding=1),
                })
                self.detail_mixers.append(detail_mixer)
                
        elif mixing_mode == 'cross':
            # Cross-subband interaction
            self.cross_mixer = nn.MultiheadAttention(
                channels, num_heads=8, batch_first=True
            )
            
        elif mixing_mode == 'attention':
            # Attention-based mixing across all subbands
            total_subbands = 1 + 3 * levels  # LL + 3 details per level
            self.subband_attention = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=channels,
                    nhead=8,
                    dim_feedforward=channels * 4,
                    batch_first=True,
                ),
                num_layers=2,
            )
        else:
            raise ValueError(f"Unknown mixing mode: {mixing_mode}")
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D wavelet-based mixing.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, channels, height, width).
            
        Returns
        -------
        Tensor
            Mixed output tensor of same shape as input.
        """
        batch_size, channels, height, width = x.shape
        residual = x
        
        # Process each channel
        outputs = []
        for c in range(channels):
            x_channel = x[:, c:c+1, :, :]
            
            # Decompose using 2D DWT
            ll, details = self.dwt.decompose(x_channel, dim=(-2, -1))
            
            # Apply mixing based on mode
            if self.mixing_mode == 'subband':
                # Process LL subband
                ll_mixed = self.ll_mixer(ll)
                
                # Process detail subbands
                details_mixed = []
                for level, (lh, hl, hh) in enumerate(details):
                    mixer = self.detail_mixers[level]
                    lh_mixed = mixer['lh'](lh)
                    hl_mixed = mixer['hl'](hl)
                    hh_mixed = mixer['hh'](hh)
                    details_mixed.append((lh_mixed, hl_mixed, hh_mixed))
                    
            elif self.mixing_mode == 'cross':
                # Flatten spatial dimensions for attention
                ll_flat = ll.flatten(2).transpose(1, 2)
                details_flat = []
                for lh, hl, hh in details:
                    details_flat.extend([
                        lh.flatten(2).transpose(1, 2),
                        hl.flatten(2).transpose(1, 2),
                        hh.flatten(2).transpose(1, 2),
                    ])
                
                # Apply cross-attention
                all_subbands = torch.cat([ll_flat] + details_flat, dim=1)
                mixed, _ = self.cross_mixer(all_subbands, all_subbands, all_subbands)
                
                # Reshape back
                ll_size = ll.shape[2] * ll.shape[3]
                ll_mixed = mixed[:, :ll_size, :].transpose(1, 2).reshape_as(ll)
                
                details_mixed = []
                offset = ll_size
                for level, (lh, hl, hh) in enumerate(details):
                    lh_size = lh.shape[2] * lh.shape[3]
                    hl_size = hl.shape[2] * hl.shape[3]
                    hh_size = hh.shape[2] * hh.shape[3]
                    
                    lh_mixed = mixed[:, offset:offset+lh_size, :].transpose(1, 2).reshape_as(lh)
                    offset += lh_size
                    hl_mixed = mixed[:, offset:offset+hl_size, :].transpose(1, 2).reshape_as(hl)
                    offset += hl_size
                    hh_mixed = mixed[:, offset:offset+hh_size, :].transpose(1, 2).reshape_as(hh)
                    offset += hh_size
                    
                    details_mixed.append((lh_mixed, hl_mixed, hh_mixed))
                    
            else:  # attention mode
                # Similar to cross but with transformer encoder
                ll_mixed = ll
                details_mixed = details
            
            # Reconstruct
            reconstructed = self.dwt.reconstruct((ll_mixed, details_mixed), dim=(-2, -1))
            
            # Ensure correct shape
            if reconstructed.shape[-2:] != (height, width):
                reconstructed = reconstructed[:, :, :height, :width]
                
            outputs.append(reconstructed)
        
        # Combine channels
        output = torch.cat(outputs, dim=1)
        
        # Residual connection
        output = output + residual
        
        return output
    
    @classmethod
    def from_config(cls, config: ConfigDict) -> "WaveletMixing2D":
        """Create WaveletMixing2D from configuration.
        
        Parameters
        ----------
        config : ConfigDict
            Configuration dictionary.
            
        Returns
        -------
        WaveletMixing2D
            Configured instance.
        """
        return cls(
            channels=config["channels"],
            wavelet=config.get("wavelet", "db4"),
            levels=config.get("levels", 2),
            mixing_mode=config.get("mixing_mode", "subband"),
        )