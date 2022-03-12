from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Region:
    row: int
    col: int
    channel: int

    size: int
    attention: float


Regions = List[Region]
Contrastive = Tuple[Regions, Regions]


class Sampler:
    def __init__(self, n: int = 10) -> None:
        self.n = n

    @abstractmethod
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        pass


class UniformSampler(Sampler):
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        return torch.randint(low=0, high=values.shape[-1], size=(values.shape[0], 2, values.shape[1], self.n))


class ProbabilisticSampler(Sampler):
    def __init__(self, n: int = 10, alpha: float = 1) -> None:
        super().__init__(n)
        self.alpha = alpha

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        maxes = values.max(dim=-1, keepdim=True).values
        normalised_values = values / maxes
        
        exponentiated_values = normalised_values ** self.alpha

        sums = exponentiated_values.sum(dim=-1, keepdim=True)
        probabilities_positives = exponentiated_values / sums
        probabilities_negatives = (1 - probabilities_positives) / (probabilities_positives.shape[-1] - 1)

        indices_positives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_positives
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_negatives
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class ProbabilisticSentinelSampler(Sampler):
    def __init__(self, n: int = 10, alpha: float = 1) -> None:
        super().__init__(n)
        self.alpha = alpha

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        sentinels_added = torch.dstack((values, torch.ones(values.shape[:-1], device=values.device)))

        exponentiated_values = sentinels_added ** self.alpha

        sums = exponentiated_values.sum(dim=-1, keepdim=True)
        probabilities_positives = exponentiated_values / sums
        probabilities_negatives = (1 - probabilities_positives) / (probabilities_positives.shape[-1] - 1)

        indices_positives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_positives
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_negatives
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class TopKSampler(Sampler):
    def __init__(self, n: int = 10, k: int = 50) -> None:
        super().__init__(n)
        self.k = k
    
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        top_k_positive_indices = torch.topk(values, k=self.k, dim=-1).indices
        top_k_negative_indices = torch.topk(-values, k=values.shape[-1] - self.k, dim=-1).indices

        indices_positives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n,))]
                        for channel in image
                    ]
                )
                for image in top_k_positive_indices
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n,))]
                        for channel in image
                    ]
                )
                for image in top_k_negative_indices
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)
