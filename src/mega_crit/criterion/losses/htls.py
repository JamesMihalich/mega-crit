import torch
from torch import nn
import numpy as np


class HTLS(nn.Module):
    def __init__(
        self,
        gamma: np.ndarray,
        w: float,
        iters: int = 2,
        sigma: int = 1,
    ):
        super().__init__()

        self.gamma = gamma
        self.w = w
        self.iters = iters
        self.sigma = sigma

        # define which img features are possible and validate arg identifies one

    def axis_adjacent_interaction_indicator(self, x: torch.Tensor) -> torch.Tensor:

        C, H, W, D = x.shape

        indices = np.indices((H, W, D)).reshape(3, -1).T
        print(indices.shape)

        print(indices[:, None, :].shape)
        print(indices[None, :, :].shape)

        delta = np.abs(indices[:, None, :] - indices[None, :, :])

        print(delta.shape)

        a = (delta.sum(axis=-1) == 1).astype(int)

        a = torch.from_numpy(a)

        return a

    def img_feature_similarity(self, img_feature: str, img: torch.Tensor) -> torch.Tensor:
        # encourage mixing when image features are similar, discourage when image features are different

        channel_count = img.shape[0]

        v = img.float().reshape(channel_count, -1)

        D = v[:, :, None] - v[:, None, :]
        l2_norm = torch.sum(D**2, dim=0)

        # identical vectors have similarity 1, different vectors have similarity increasingly approch 0
        return torch.exp(-l2_norm / self.sigma**2)

    def label_interaction_matrix(self, lab: torch.Tensor):
        pass

    def smooth_label(self, img: torch.Tensor, lab: torch.Tensor):

        # gather adjacency rules
        a = self.axis_adjacent_interaction_indicator(img)

        # gather image similarity weights
        S = self.img_feature_similarity("intensity", img)

        # build edge weights
        A_y = a * S

        # build the graph lapacian
        L_y = torch.diag(A_y.sum(dim=1)) - A_y

        # put the label tensor on device
        Y = lab.to(device=S.device, dtype=S.dtype).reshape(lab.shape[0], -1)
        U = Y.T

        for _ in range(self.iters):
            U = U - self.w * (L_y @ U)

        return U.T.reshape(lab.shape)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        pass


if __name__ == "__main__":
    tensor = torch.rand([2, 4, 4, 4])
    gamma = np.ones((4, 4))

    htls = HTLS(gamma, iters=2, sigma=1, w=(1 / 6))

    # gaussian_similarity = htls.img_feature_similarity(img_feature="intensity", img=tensor)
    # print(gaussian_similarity.shape)

    htls.smooth_label(tensor, tensor)
