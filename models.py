import torch
import torch.nn as nn


class ModelBuilder():
    def build_encoder(self, weights=''):
        net_encoder = VGGEncoder()
        if len(weights) > 0:
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_encoder

    def build_decoder(self, num_class=19, use_softmax=True, weights=''):
        net_decoder = VGGDecoder(num_class, use_softmax)
        if len(weights) > 0:
            pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage)
            net_decoder.load_state_dict(pretrained_dict, strict=False)
        return net_decoder


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, pool1_idx, pool2_idx, pool3_idx


class VGGDecoder(nn.Module):
    def __init__(self, num_class, use_softmax):
        super(VGGDecoder, self).__init__()
        self.num_class = num_class
        self.use_softmax = use_softmax

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_out = nn.Conv2d(64, self.num_class, 3, 1, 0)

    def forward(self, x, pool1_idx=None, pool2_idx=None, pool3_idx=None):
        out = x

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.unpool3(out, pool3_idx)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        out = self.relu3_4(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.unpool2(out, pool2_idx)

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.unpool1(out, pool1_idx)

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)

        out = self.pad1_1(out)
        out = self.conv_out(out)

        if self.use_softmax:
            out = nn.functional.log_softmax(out, dim=1)

        return out
