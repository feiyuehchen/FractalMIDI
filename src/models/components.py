import torch
import torch.nn as nn
import math

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Conditioning vector modulates the scale and shift of LayerNorm.
    """
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, embed_dim * 2)
        
        # Initialize projection to be identity-like
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias[:embed_dim].fill_(1.0)
            self.proj.bias[embed_dim:].fill_(0.0)

    def forward(self, x, cond):
        # x: (B, T, E)
        # cond: (B, C) or (B, 1, C)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1) # (B, 1, C)
            
        scale_shift = self.proj(cond) # (B, 1, 2*E)
        scale, shift = scale_shift.chunk(2, dim=-1) # (B, 1, E)
        
        x = self.norm(x)
        x = x * scale + shift
        return x

class AdaLNModulator(nn.Module):
    """
    AdaLN Modulator only (Scale & Shift).
    Assumes input is already normalized.
    """
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, embed_dim * 2)
        # Zero-init to start as Identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias) 
        
    def forward(self, x, cond):
        # x: (B, T, E)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
            
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Modulation: x * (1 + scale) + shift
        x = x * (1 + scale) + shift
        return x

class TimePositionalEmbedding(nn.Module):
    """
    Sinusoidal Time Positional Embedding.
    """
    def __init__(self, embed_dim, max_len=16384):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, E)
        T = x.size(1)
        return self.pe[:T, :].unsqueeze(0) # (1, T, E)

class FractalInputProj(nn.Module):
    """
    Project Piano Roll (B, 2, T, 128) -> (B, T, E).
    Uses Conv2d over Pitch axis to capture harmonic relationships (intervals) invariantly.
    Includes Bar Position Embedding.
    """
    def __init__(self, in_channels, embed_dim, max_bar_len=16):
        super().__init__()
        # Input: (B, 2, T, 128)
        
        self.pitch_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(12, 1), stride=(1, 1), padding=(6, 0)),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(12, 1), stride=(1, 1), padding=(5, 0)), 
            nn.GELU(),
        )
        
        self.proj = nn.Conv1d(32 * 128, embed_dim, kernel_size=1)
        
        self.pos_emb = TimePositionalEmbedding(embed_dim)
        self.bar_emb = nn.Embedding(max_bar_len, embed_dim)

    def forward(self, x, bar_pos=None):
        x = x.permute(0, 1, 3, 2) 
        x = self.pitch_conv(x) # (B, 32, 128, T)
        B, C, P, T = x.shape
        x = x.reshape(B, C * P, T)
        x = self.proj(x) # (B, E, T)
        x = x.permute(0, 2, 1) # (B, T, E)
        
        x = x + self.pos_emb(x)
        
        if bar_pos is not None:
            # Clamp bar_pos to valid range to prevent crashes
            max_valid_idx = self.bar_emb.num_embeddings - 1
            bar_pos_clamped = torch.clamp(bar_pos, 0, max_valid_idx)
            x = x + self.bar_emb(bar_pos_clamped)
            
        return x

class FractalOutputProj(nn.Module):
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels * 128)

    def forward(self, x):
        B, T, E = x.shape
        x = self.proj(x) 
        x = x.view(B, T, 2, 128)
        x = x.permute(0, 2, 1, 3) 
        return x

class LSTMOutputProj(nn.Module):
    """
    LSTM-based Decoder for Pitch-wise Autoregression.
    
    Structure:
    - Condition: Transformer Embedding (B, T, E)
    - Input: Previous Pitch Token (B, T, 2)
    - LSTM: 2 Layers, Hidden 1024
    - Output: Current Pitch Token (B, T, 2) (Note On/Off logits + Velocity)
    """
    def __init__(self, embed_dim, out_channels=2, hidden_dim=1024, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project Transformer embedding to initialize LSTM state or as constant input
        # Here we concatenate embedding with input at each step
        self.cond_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Input projection (Pitch Token -> Hidden)
        # Pitch Token is (Note, Velocity)
        self.input_proj = nn.Linear(out_channels, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2, # Input + Condition
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_head = nn.Linear(hidden_dim, out_channels)
        
        # SOS Token for Pitch Sequence (learned)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.normal_(self.sos_token, std=0.02)

    def forward(self, x, target=None, teacher_forcing=True):
        """
        Args:
            x: Transformer Embeddings (B, T, E)
            target: Ground Truth Piano Roll (B, 2, T, 128) - optional for inference
            teacher_forcing: Whether to use teacher forcing (training) or autoregressive (inference)
        
        Returns:
            logits: (B, 2, T, 128)
        """
        B, T, E = x.shape
        
        # 1. Prepare Condition
        # (B, T, E) -> (B*T, 1, E) -> (B*T, 1, H)
        cond = self.cond_proj(x).reshape(B*T, 1, self.hidden_dim)
        
        if teacher_forcing and target is not None:
            # Training Mode: Parallel Pitch Generation
            # Target: (B, 2, T, 128) -> (B, T, 2, 128) -> (B, T, 128, 2) -> (B*T, 128, 2)
            target_seq = target.permute(0, 2, 3, 1).reshape(B*T, 128, self.out_channels)
            
            # Prepare Inputs: [SOS, p_0, ..., p_126]
            sos = self.sos_token.expand(B*T, 1, -1)
            lstm_input_seq = torch.cat([sos, target_seq[:, :-1, :]], dim=1) # (B*T, 128, 2)
            
            # Embed Inputs
            lstm_input_emb = self.input_proj(lstm_input_seq) # (B*T, 128, H)
            
            # Concatenate Condition to every step
            # cond: (B*T, 1, H) -> expand to (B*T, 128, H)
            cond_expanded = cond.expand(-1, 128, -1)
            
            full_input = torch.cat([lstm_input_emb, cond_expanded], dim=-1) # (B*T, 128, 2*H)
            
            # LSTM Forward
            output, _ = self.lstm(full_input) # (B*T, 128, H)
            
            # Project to Output
            logits = self.output_head(output) # (B*T, 128, 2)
            
            # Reshape back: (B, T, 128, 2) -> (B, 2, T, 128)
            logits = logits.view(B, T, 128, 2).permute(0, 3, 1, 2)
            
            return logits
            
        else:
            # Inference Mode: Autoregressive Pitch Generation
            # We generate 128 pitches sequentially for each time step
            # x is (B, T, E)
            
            # Prepare initial state (default zeros)
            h_state = None 
            
            # Prepare Condition: (B*T, 1, H)
            cond_step = cond # Constant for all pitch steps
            
            # Initial Input: SOS (B*T, 1, 2)
            curr_input = self.sos_token.expand(B*T, 1, -1)
            
            outputs = []
            
            for p in range(128):
                # Embed Input
                curr_emb = self.input_proj(curr_input) # (B*T, 1, H)
                
                # Concat Condition
                full_input = torch.cat([curr_emb, cond_step], dim=-1) # (B*T, 1, 2*H)
                
                # LSTM Step
                out_step, h_state = self.lstm(full_input, h_state) # out: (B*T, 1, H)
                
                # Logits
                logit_step = self.output_head(out_step) # (B*T, 1, 2)
                outputs.append(logit_step)
                
                # Sample for next input (Greedy or Sampling?)
                # For simplicity in this module, we use greedy/logits as input?
                # Ideally we should sample. But here we just return logits.
                # To make this module self-contained for "forward", we need to sample.
                # BUT, usually 'forward' implies training. 'sample' implies inference.
                # However, we are called from TemporalFractalNetwork.sample/forward.
                
                # If we are just forwarding without target (inference), we MUST sample to get next input.
                # Let's assume Greedy for simple forwarding, or use probabilistic sampling if needed.
                # For now: Sigmoid > 0.5 for Note, Identity for Vel?
                
                # But wait, to be consistent with external sampling control (temp etc),
                # maybe we should expose a 'sample_pitch' method?
                # Or just use the prediction as next input (Teacher Forcing with self-prediction).
                
                pred_note = torch.sigmoid(logit_step[:, :, 0]) > 0.5
                pred_vel = torch.clamp(logit_step[:, :, 1], 0, 1)
                curr_input = torch.stack([pred_note.float(), pred_vel], dim=-1)

            # Stack outputs
            logits = torch.cat(outputs, dim=1) # (B*T, 128, 2)
            logits = logits.view(B, T, 128, 2).permute(0, 3, 1, 2)
            
            return logits

    def sample(self, x, temperature=1.0):
        """
        Explicit sampling method for inference.
        """
        B, T, E = x.shape
        cond = self.cond_proj(x).reshape(B*T, 1, self.hidden_dim)
        
        h_state = None
        curr_input = self.sos_token.expand(B*T, 1, -1)
        outputs = []
        
        for p in range(128):
            curr_emb = self.input_proj(curr_input)
            full_input = torch.cat([curr_emb, cond], dim=-1)
            out_step, h_state = self.lstm(full_input, h_state)
            logit_step = self.output_head(out_step)
            
            # Store Logit
            outputs.append(logit_step)
            
            # Sample next input
            logits_note = logit_step[:, :, 0]
            logits_vel = logit_step[:, :, 1]
            
            if temperature > 0:
                probs = torch.sigmoid(logits_note / temperature)
                pred_note = torch.bernoulli(probs)
            else:
                pred_note = (logits_note > 0).float()
                
            pred_vel = torch.clamp(logits_vel, 0, 1)
            curr_input = torch.stack([pred_note, pred_vel], dim=-1)
            
        logits = torch.cat(outputs, dim=1) # (B*T, 128, 2)
        logits = logits.view(B, T, 128, 2).permute(0, 3, 1, 2)
        return logits

class StructureInputProj(nn.Module):
    def __init__(self, in_channels, embed_dim, max_bar_len=16):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.pos_emb = TimePositionalEmbedding(embed_dim)
        self.bar_emb = nn.Embedding(max_bar_len, embed_dim)

    def forward(self, x, bar_pos=None):
        x = x.permute(0, 2, 1) # (B, T, 2)
        x = self.proj(x) # (B, T, E)
        x = x + self.pos_emb(x)
        
        if bar_pos is not None:
            # Clamp bar_pos to valid range to prevent crashes
            max_valid_idx = self.bar_emb.num_embeddings - 1
            bar_pos_clamped = torch.clamp(bar_pos, 0, max_valid_idx)
            x = x + self.bar_emb(bar_pos_clamped)
            
        return x

class StructureOutputProj(nn.Module):
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        x = self.proj(x) 
        x = x.permute(0, 2, 1) 
        return x

