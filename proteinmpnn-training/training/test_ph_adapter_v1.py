#!/usr/bin/env python3
"""Small deterministic tests for the continuous-pH ProteinMPNN adapter."""

from __future__ import annotations

import unittest

import torch

from model_utils_ph_adapter_v1 import PHConditionedProteinMPNN, freeze_source_model


class PHAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PHConditionedProteinMPNN(
            num_letters=27,
            vocab=27,
            node_features=16,
            edge_features=16,
            hidden_dim=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            k_neighbors=4,
            augment_eps=0.0,
            dropout=0.0,
            ph_num_rbf=8,
        )
        self.model.eval()
        self.X = torch.randn(2, 7, 4, 3)
        self.S = torch.randint(0, 27, (2, 7))
        self.mask = torch.ones(2, 7)
        self.chain_M = torch.ones(2, 7)
        self.residue_idx = torch.arange(7).repeat(2, 1)
        self.chain_encoding = torch.ones(2, 7)

    def run_model(self, ph: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(99)
        return self.model(
            self.X,
            self.S,
            self.mask,
            self.chain_M,
            self.residue_idx,
            self.chain_encoding,
            ph,
        )

    def test_zero_initialization_is_exactly_ph_invariant(self) -> None:
        low = self.run_model(torch.tensor([4.0, 4.0]))
        high = self.run_model(torch.tensor([10.0, 10.0]))
        self.assertEqual(float((low - high).abs().max().detach()), 0.0)

    def test_freeze_source_model_leaves_only_ph_parameters_trainable(self) -> None:
        trainable = freeze_source_model(self.model)
        self.assertGreater(sum(parameter.numel() for parameter in trainable), 0)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("ph_"), name)

    def test_adapter_receives_gradient(self) -> None:
        freeze_source_model(self.model)
        self.model.train()
        output = self.run_model(torch.tensor([4.0, 10.0]))
        loss = -output[..., 0].mean()
        loss.backward()
        final_layers = (
            self.model.ph_hidden_net[-1].weight,
            self.model.ph_state_net[-1].weight,
        )
        self.assertTrue(all(parameter.grad is not None for parameter in final_layers))
        self.assertTrue(any(float(parameter.grad.abs().sum()) > 0 for parameter in final_layers))

    def test_invalid_ph_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.ph_features(torch.tensor([-0.1]))
        with self.assertRaises(ValueError):
            self.model.ph_features(torch.tensor([float("nan")]))


if __name__ == "__main__":
    unittest.main()
