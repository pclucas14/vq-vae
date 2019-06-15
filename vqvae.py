import torch
from torch import nn
from torch.nn import functional as F


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
# Taken    from https://github.com/rosinality/vq-vae-2-pytorch

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 1: 
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, padding=2), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(channel // 2, channel // 2, 3, padding=1)
            ]

        for i in range(n_res_block):
            blocks += [ResBlock(channel, n_res_channel)]

        blocks += [nn.ReLU(inplace=True)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks += [ResBlock(channel, n_res_channel)]

        blocks += [nn.ReLU(inplace=True)]

        if stride == 8:
            blocks += [
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        if stride == 4:
            blocks += [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        elif stride == 2:
            blocks += [nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)]
        
        elif stride == 1: 
            blocks += [nn.Conv2d(channel, out_channel, 3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel = args.input_size[0]
        channel    = args.n_channels
        n_res_block = args.n_res_blocks
        n_res_channel = args.n_res_channels
        embed_dim = args.embed_dim
        n_embed = args.n_embeds
        decay = args.decay

        assert embed_dim % args.n_codebooks == 0, 'you need that last dimension to be evenly divisible by the amt of codebooks'

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=args.downsample)
        
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=args.downsample)

        # build the codebooks
        self.quantize = nn.ModuleList([Quantize(embed_dim // args.n_codebooks, n_embed) for _ in range(args.n_codebooks)])
        
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))
    
    def forward(self, input):
        z_q, diff, _ = self.encode(input)
        dec = self.decode(z_q)

        return dec, diff

    def encode(self, input):
        pre_z_e = self.enc(input)
        z_e     = self.quantize_conv(pre_z_e)

        # divide into multiple codebooks
        z_e_s   = z_e.chunk(len(self.quantize), 1)

        z_q_s, argmins = [], []
        diffs = 0.
        
        for z_e, quantize in zip(z_e_s, self.quantize):
            z_q, diff, argmin = quantize(z_e.permute(0, 2, 3, 1))
            z_q_s   += [z_q]
            argmins += [argmin]
            diffs   += diff
        
        z_q = torch.cat(z_q_s, dim=-1)
        z_q = z_q.permute(0, 3, 1, 2)
        
        return z_q, diffs, argmins

    def decode(self, quant):
        return self.dec(quant)



class VQVAE_2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=2) #4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=2, #4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_b = self.quantize_b.embed_code(code_b)

        dec = self.decode(quant_t, quant_b)

        return dec
