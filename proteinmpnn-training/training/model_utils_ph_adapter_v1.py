"""Zero-initialized continuous-pH adapter for the 27-state ProteinMPNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import ProteinMPNN, cat_neighbors_nodes, gather_nodes


ALPHABET = "ACDEFGHIKLMNPQRSTVWYcdhekrX"
STATE_PAIRS = (("C", "c"), ("D", "d"), ("E", "e"), ("H", "h"), ("K", "k"), ("R", "r"))


class PHConditionedProteinMPNN(ProteinMPNN):
    """ProteinMPNN with a small global pH FiLM/state-odds adapter.

    Only the new ``ph_*`` parameters are trained in the lightweight pilot.  The
    final layers are initialized to zero, making the model exactly identical to
    its source checkpoint before the first update for every pH value.
    """

    def __init__(self, *args, ph_num_rbf: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.ph_num_rbf = int(ph_num_rbf)
        centers = torch.linspace(0.0, 14.0, self.ph_num_rbf)
        self.register_buffer("ph_rbf_centers", centers)
        width = 14.0 / max(self.ph_num_rbf - 1, 1)
        self.register_buffer("ph_rbf_width", torch.tensor(width))
        feature_dim = self.ph_num_rbf + 1

        self.ph_hidden_net = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.ph_state_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, len(STATE_PAIRS)),
        )

        projection = torch.zeros(len(STATE_PAIRS), len(ALPHABET))
        for pair_index, (canonical, alternate) in enumerate(STATE_PAIRS):
            projection[pair_index, ALPHABET.index(canonical)] = -0.5
            projection[pair_index, ALPHABET.index(alternate)] = 0.5
        self.register_buffer("ph_state_projection", projection)

        # Identity-preserving initialization: source checkpoint outputs are
        # bitwise unchanged apart from ordinary floating-point execution order.
        nn.init.zeros_(self.ph_hidden_net[-1].weight)
        nn.init.zeros_(self.ph_hidden_net[-1].bias)
        nn.init.zeros_(self.ph_state_net[-1].weight)
        nn.init.zeros_(self.ph_state_net[-1].bias)

    def ph_features(self, ph: torch.Tensor) -> torch.Tensor:
        ph = ph.to(dtype=self.ph_rbf_centers.dtype).reshape(-1)
        if not torch.isfinite(ph).all():
            raise ValueError("pH contains a non-finite value")
        if ((ph < 0.0) | (ph > 14.0)).any():
            raise ValueError("pH must lie in [0, 14]")
        rbf = torch.exp(
            -0.5
            * ((ph[:, None] - self.ph_rbf_centers[None, :]) / self.ph_rbf_width) ** 2
        )
        normalized = ((ph - 7.0) / 7.0)[:, None]
        return torch.cat((rbf, normalized), dim=-1)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        ph: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        ph_features = self.ph_features(ph)
        ph_hidden = self.ph_hidden_net(ph_features)
        h_V = h_V + mask.unsqueeze(-1) * ph_hidden[:, None, :]

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * torch.abs(torch.randn(chain_M.shape, device=device))
        )
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij,biq,bjp->bqp",
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1d = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1d * mask_attend
        mask_fw = mask_1d * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        state_shift = self.ph_state_net(ph_features)
        token_bias = state_shift @ self.ph_state_projection
        logits = logits + token_bias[:, None, :]
        log_probs = F.log_softmax(logits, dim=-1)
        if return_logits:
            return log_probs, logits
        return log_probs


def freeze_source_model(model: PHConditionedProteinMPNN) -> list[nn.Parameter]:
    """Freeze the source model and return only adapter parameters."""
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("ph_")
        if parameter.requires_grad:
            trainable.append(parameter)
    return trainable

