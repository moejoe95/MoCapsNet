########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def squash(s, dim=-1):
    '''
    "Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||

    Args:
            s: 	Vector before activation
            dim:	Dimension along which to calculate the norm

    Returns:
            Squashed vector
    '''
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps,
                 kernel_size=9, stride=2, padding=0):
        """
        Initialize the layer.

        Args:
                in_channels: 	Number of input channels.
                out_channels: 	Number of output channels.
                dim_caps:		Dimensionality, i.e. length, of the output capsule vector.

        """
        super(PrimaryCapsules, self).__init__()
        self.dim_caps = dim_caps
        self._caps_channel = int(out_channels / dim_caps)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), self._caps_channel,
                       out.size(2), out.size(3), self.dim_caps)
        out = out.view(out.size(0), -1, self.dim_caps)
        return squash(out)


class RoutingCapsules(nn.Module):
    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device: torch.device, routing='RBA'):
        """
        Initialize the layer.

        Args:
                in_dim: 		Dimensionality (i.e. length) of each capsule vector.
                in_caps: 		Number of input capsules if digits layer.
                num_caps: 		Number of capsules in the capsule layer
                dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
                num_routing:	Number of iterations during routing algorithm
        """
        super(RoutingCapsules, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device
        self.routing = routing.upper()

        self.W = nn.Parameter(
            0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim))
        if routing.upper() == "SDA":
            self.bias = nn.Parameter(torch.empty(1, num_caps, dim_caps))
            nn.init.constant_(self.bias, 0.1)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = ' -> '
        res = self.__class__.__name__ + '('
        res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
        res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
        res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
        res = res + 'num_routing=' + str(self.num_routing) + ')'
        res = res + line + ')'
        return res

    def forward(self, x):
        if self.routing == "RBA":
            return self.dynamic_routing(x)
        elif self.routing == "SDA":
            return self.sda_routing(x)
        else:
            raise NotImplementedError(
                "No routing algorithm with this name found.")

    def dynamic_routing(self, x):
        """
        Dynamic routing by Sabour et al.
        """
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        '''
		Procedure 1: Routing algorithm
		'''
        b = torch.zeros(batch_size, self.num_caps,
                        self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing-1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = F.softmax(b, dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)

        # apply "squashing" non-linearity along dim_caps
        return squash(s)

    def sda_routing(self, u):
        """
        Scaled-distance-agreement routing by Peer et al.
        """
        batch_size = u.size(0)
        u_norm = torch.norm(u, dim=-1)

        u = torch.unsqueeze(torch.unsqueeze(u, 1), 3)
        u = torch.tile(u, [1, self.num_caps, 1, 1, 1])
        u = torch.tile(u, [1, 1, 1, self.dim_caps, 1])

        # tile over batch size
        w = torch.tile(self.W, [batch_size, 1, 1, 1, 1])

        # Dotwise product between u and w to get all votes
        u_hat = torch.sum(u * w, dim=-1)

        # ensure that ||u_hat|| <= ||v_i||
        u_hat = self.restrict_prediction(u_hat, u_norm)

        # scaled distance agreement routing
        bias = torch.tile(self.bias, [batch_size, 1, 1])

        # detach to prevent gradient flow while routing
        temp_u_hat = u_hat.detach()
        temp_bias = bias.detach()

        b_ij = torch.zeros(batch_size, self.num_caps,
                           self.in_caps, 1, requires_grad=False).to(self.device)

        for r in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij_tiled = torch.tile(c_ij, [1, 1, 1, self.dim_caps])
            if r == self.num_routing - 1:
                # use u_hat and bias in the last iteration for grad.
                s_j = torch.sum(c_ij_tiled * u_hat, dim=2) + bias
            else:
                s_j = torch.sum(c_ij_tiled * temp_u_hat, dim=2) + temp_bias
            v_j = squash(s_j)

            if r < self.num_routing - 1:
                v_j = torch.unsqueeze(v_j, 2)
                v_j = torch.tile(v_j, [1, 1, self.in_caps, 1])

                # calculate scale factor t
                p_p = 0.9
                d = torch.norm(v_j - temp_u_hat, dim=-1, keepdim=True)
                d_o = torch.mean(torch.mean(d)).item()
                d_p = d_o * 0.5
                t = np.log(p_p * (self.num_caps - 1)) - \
                    np.log(1 - p_p) / (d_p - d_o + 1e-12)

                b_ij = t * d

        return v_j

    def restrict_prediction(self, u_hat, u_norm):
        u_hat_norm = torch.norm(u_hat, dim=-1, keepdim=True)
        u_norm = torch.unsqueeze(torch.unsqueeze(u_norm, 1), 3)
        u_norm = torch.tile(u_norm, [1, self.num_caps, 1, self.dim_caps])
        new_u_hat_norm = torch.minimum(u_hat_norm, u_norm)
        return u_hat / u_hat_norm * new_u_hat_norm
