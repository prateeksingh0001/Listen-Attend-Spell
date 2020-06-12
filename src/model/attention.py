import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: Add Scaled dot product attention
class Attention(nn.Module):
    def __init__(self, mode="scaled_dot"):
        super().__init__()

        if mode in ("scaled_dot", "dot"):
            self.mode = mode
        else:
            raise NotImplementedError("Attention mode not implemented")

    def forward(self, query, key, value, mask_lengths):
        """
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (T, N, key_size) Key Projection from Encoder per time step
        :param value: (T, N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        """

        key = torch.transpose(key, 0, 1)
        value = torch.transpose(value, 0, 1)

        attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        if self.mode == "scaled_dot":
            key_dims = key.shape[2]
            scale_factor = float(key_dims)**(-1/2)
            attention = attention * scale_factor

        mask = torch.arange(key.size(1), device=DEVICE).unsqueeze(0) >= mask_lengths.unsqueeze(1).to(DEVICE)
        mask.to(DEVICE)

        attention.masked_fill_(mask, -1e9)
        attention = attention.softmax(dim=1)

        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return context, attention
