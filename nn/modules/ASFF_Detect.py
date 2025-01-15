import torch
import torch.nn as nn
from ultralytics.utils.tal import  dist2bbox, make_anchors
import math
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=(k - 1) // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).float()
        x = torch.arange(c1).float().view(1, -1).float()
        self.register_buffer("dfl_weight", x)

    def forward(self, x):
        b, c, h, w = x.shape
        return (x.view(b, c, h * w).softmax(1) @ self.dfl_weight.to(x.dtype).view(c, 1)).view(b, 1, h, w)


class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=True, vis=False, act_cfg=True):
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier), int(128 * multiplier)]  # 更新维度列表
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)  # 处理新增层
            self.expand = Conv(self.inter_dim, self.inter_dim, 3, 1)  # 修改expand输出维度
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)  # 处理新增层
            self.expand = Conv(self.inter_dim, self.inter_dim, 3, 1)  # 修改expand输出维度
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)  # 处理新增层
            self.expand = Conv(self.inter_dim, self.inter_dim, 3, 1)  # 修改expand输出维度
        elif level == 3:  # 处理新增的层级
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, self.inter_dim, 3, 1)  # 修改expand输出维度
        compress_c = 8 if rfb else 16  # 修改compress_c
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(
            self.inter_dim, compress_c, 1, 1)  # 新增层的权重
        self.weight_levels = Conv(
            compress_c * 4, 4, 1, 1)  # 更新输出通道数
        self.vis = vis

    def forward(self, x):  # l, m, s, xs
        x_level_0 = x[3]  # l
        x_level_1 = x[2]  # m
        x_level_2 = x[1]  # s
        x_level_3 = x[0]  # xs # 新增的最小的特征图
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            level_3_resized = self.stride_level_3(x_level_3)  # 处理新增层

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)  # 处理新增层
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)  # 处理新增层
        elif self.level == 3:  # 处理新增的层级
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=8, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=4, mode='nearest')
            x_level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(
                x_level_2_compressed, scale_factor=2, mode='nearest')
            level_3_resized = x_level_3

        # 在计算权重之前，将所有特征图都调整为与 level_3_resized 相同的大小
        level_0_resized = F.interpolate(level_0_resized, size=level_3_resized.shape[2:], mode='nearest')
        level_1_resized = F.interpolate(level_1_resized, size=level_3_resized.shape[2:], mode='nearest')
        level_2_resized = F.interpolate(level_2_resized, size=level_3_resized.shape[2:], mode='nearest')

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)  # 新增层的权重
        print(f"level_0_weight_v shape: {level_0_weight_v.shape}")
        print(f"level_1_weight_v shape: {level_1_weight_v.shape}")
        print(f"level_2_weight_v shape: {level_2_weight_v.shape}")
        print(f"level_3_weight_v shape: {level_3_weight_v.shape}")
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]  # 更新融合层
        out = self.expand(fused_out_reduced)
        return out

class ASFF_Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), multiplier=0.25, rfb=False):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # 更新通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # 新增 ASFF 模块
        self.l0_fusion = ASFFV5(level=0, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = ASFFV5(level=1, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, multiplier=multiplier, rfb=rfb)
        self.l3_fusion = ASFFV5(level=3, multiplier=multiplier, rfb=rfb) # 新增的ASFF模块

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # 使用新的ASFF模块
        x1 = self.l0_fusion(x)
        x2 = self.l1_fusion(x)
        x3 = self.l2_fusion(x)
        x4 = self.l3_fusion(x) # 使用新的ASFF模块
        print(f"x1 shape:{x1.shape}")
        print(f"x2 shape:{x2.shape}")
        print(f"x3 shape:{x3.shape}")
        print(f"x4 shape:{x4.shape}")
        x = [x4, x3, x2, x1] # 注意顺序，最小的特征图放到最前面
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        if self.export and self.format in ('tflite', 'edgetpu'):
            # integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

if __name__ == "__main__":

    # Generating Sample image
    image1 = (1, 32, 64, 64)
    image2 = (1, 64, 32, 32)
    image3 = (1, 128, 16, 16)
    image4 = (1, 256, 8, 8)

    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image4 = torch.rand(image4)
    image = [image1, image2, image3, image4]
    channel = (32, 64, 128, 256)
    # Model
    mobilenet_v1 = ASFF_Detect(nc=80, ch=channel)

    out = mobilenet_v1(image)
    print(out)