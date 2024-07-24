import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class FPN(nn.Module):
    def __init__(self, out_channels=256, add_channels=102, lvl=3):
        super(FPN, self).__init__()
        
        lateral_convs = []
        output_convs = []
        inner_convs = []

        input_channels = [1024, 512, 256]

        for idx in range(lvl):
            lateral_conv = nn.Conv2d(out_channels * 2 + add_channels + input_channels[idx], out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            # inner_cls = nn.Sequential(
            #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            #     nn.ReLU(inplace=True), 
            #     nn.Dropout2d(0.1), 
            #     nn.Conv2d(out_channels, 2, kernel_size=1)
            # )
            inner_cls = nn.Conv2d(out_channels, 2, kernel_size=1)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            weight_init.c2_xavier_fill(inner_cls)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            inner_convs.append(inner_cls)
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)
        self.inner_convs = nn.ModuleList(inner_convs)
                
    def forward(self, q_multi, q_feat, s_feat, masks):
        s_feat = s_feat.transpose(-1,-2).unsqueeze(-1).expand_as(q_feat)

        outputs = []
        for idx, q_fpnm in enumerate(q_multi):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]

            tmp_mask = F.interpolate(masks, size=q_fpnm.shape[-2:], mode="nearest")
            tmp_q_feat = F.interpolate(q_feat, size=q_fpnm.shape[-2:], mode="bilinear", align_corners=True)
            tmp_s_feat = F.interpolate(s_feat, size=q_fpnm.shape[-2:], mode="bilinear", align_corners=True)
            feats = torch.cat([tmp_q_feat, tmp_s_feat, tmp_mask, q_fpnm], dim=1)
            cur_fpn = lateral_conv(feats)
            if idx==0:
                y = output_conv(cur_fpn)
            else:
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            outputs.append(self.inner_convs[idx](y))

        return y, outputs