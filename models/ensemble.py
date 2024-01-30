import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import elu
from .modules import Expression, Ensure4d

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

def get_fc(c_in, c_out, bias=True):
    layer = nn.Linear(c_in, c_out, bias=bias)
    init.kaiming_normal_(layer.weight.data)
    if bias:
        layer.bias.data.zero_()
    return layer

def get_classifier(ch_out, n_classes):
    net = nn.Sequential(get_fc(ch_out, n_classes))
    return net

def get_backbone_eegnetv4(k_width, in_chans, input_window_samples, n_classes, dropout_1_prob, backbone_indices=[0, 1, 2, 3]):
    net = Backbone_EEGNetv4(in_chans=in_chans,
                            input_window_samples=input_window_samples,
                            final_conv_length="auto",
                            F1=k_width*8,
                            D=2,
                            F2=k_width*16,
                            kernel_length=64,
                            n_classes=n_classes,
                            dropout_1_prob=dropout_1_prob,
                            backbone_indices=backbone_indices,
    )
    return net

class Backbone_EEGNetv4(nn.Sequential):
    def __init__(
            self,
            in_chans,
            input_window_samples=None,
            final_conv_length="auto",
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            n_classes=2,
            dropout_1_prob=0.0,
            backbone_indices=[0, 1, 2, 3],
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.n_classes = n_classes
        self.dropout_1_prob = dropout_1_prob
        self.backbone_indices = backbone_indices

        n_out_time = 12 # TODO: set this automatically, instead of hard-coded
        self.final_conv_length = n_out_time
        self._num_features = self.F2 * self.final_conv_length
        ##############################################################################

        self.modules_all = nn.Sequential()
        for module_index in self.backbone_indices:
            module_name = "module_{}".format(module_index)
            module_ = self.get_module(module_index=module_index)
            self.modules_all.add_module(module_name, module_)

    def get_module(self, module_index):

        if module_index==0:
            module_ = self.get_module_temporal()
        elif module_index==1:
            module_ = self.get_module_spatial()
        elif module_index==2:
            module_ = self.get_module_sep_1()
        elif module_index==3:
            module_ = self.get_module_sep_2()
        elif module_index==4:
            module_ = self.get_module_classifier()

        return module_

    def get_module_temporal(self):
        # Ensure4D: BxCxT   --> BxCxTx1
        # Permute:  BxCxTx1 --> Bx1xCxT
        module_ = nn.Sequential(
            Ensure4d(),
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(1, self.F1,
                      (1, self.kernel_length),
                      stride=1, bias=False,
                      padding=(0, self.kernel_length // 2)
                      ),
        )
        return module_

    def get_module_spatial(self):
        module_ = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.in_chans, 1),
                                 max_norm=1.0,
                                 stride=1,
                                 bias=False,
                                 groups=self.F1,
                                 padding=(0, 0),
            ),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.1, affine=True, eps=1e-3),
            Expression(elu),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_1_prob),
        )
        return module_

    def get_module_sep_1(self):
        module_ = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D,
                      (1, 16),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D,
                      padding=(0, 16 // 2),
            ),
        )
        return module_

    def get_module_sep_2(self):
        module_ = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F2,
                      (1, 1),
                      stride=1,
                      bias=False,
                      padding=(0, 0),
            ),
            nn.BatchNorm2d(self.F2, momentum=0.1, affine=True, eps=1e-3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
            LambdaLayer(lambda x: x.squeeze(2)),
            nn.Flatten(),
        )
        return module_

    def get_module_classifer(self):
        module_ = nn.Sequential(
            get_classifier(ch_out=self.num_features,
                           n_classes=self.n_c)
        )
        return module_

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x):
        x = self.modules_all.forward(x)
        return x

class Backbone_Shared(nn.Module):
    def __init__(self,
                 n_subsets,
                 k_width,
                 in_chans,
                 input_window_samples,
                 n_classes,
                 dropout_input_prob,
                 dropout_1_prob,
                 backbone_indices):
        super().__init__()
        self.k_width = k_width
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.n_classes = n_classes
        self.dropout_input_prob = dropout_input_prob
        self.n_subsets = n_subsets
        self.backbone_indices = backbone_indices
        self.stage_1_is_shared = True

        self.encoder = get_backbone_eegnetv4(k_width=k_width,
                                             in_chans=in_chans,
                                             input_window_samples=input_window_samples,
                                             n_classes=n_classes,
                                             dropout_1_prob=dropout_1_prob,
                                             backbone_indices=backbone_indices)
        self.num_features = self.encoder.num_features

    def forward(self, x_dict):

        if isinstance(x_dict, dict):
            x = x_dict['x']
        else:
            x = x_dict

        if self.training:
            x = nn.Dropout(self.dropout_input_prob)(x)
        feats = self.encoder(x)

        out_dict = {}
        out_dict['x'] = feats

        return out_dict

class Backbone_Multi(nn.Module):
    def __init__(self,
                 n_subsets,
                 k_width,
                 in_chans,
                 input_window_samples,
                 n_classes,
                 n_datasets,
                 dropout_input_prob,
                 dropout_1_prob,
                 backbone_indices,
                 ):
        super().__init__()
        self.n_subsets = n_subsets
        self.k_width = k_width
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.n_classes = n_classes
        self.dropout_input_prob = dropout_input_prob
        self.n_datasets_train = n_datasets
        self.backbone_indices = backbone_indices
        self.stage_1_is_shared = False

        self.encoders = nn.ModuleDict({})
        for subset_idx in range(self.n_subsets):
            encoder_name_str = 'encoder_{}'.format(str(subset_idx + 1).zfill(3))
            encoder_module = get_backbone_eegnetv4(k_width, in_chans, input_window_samples, n_classes, dropout_1_prob, backbone_indices)
            self.encoders.update({encoder_name_str: encoder_module})
            self.num_features = encoder_module.num_features

    def forward(self, x_dict):

        if isinstance(x_dict, dict):
            x = x_dict['x']
        else:
            x = x_dict

        #########################################################

        if self.training:
            x = nn.Dropout(self.dropout_input_prob)(x)

        feats_per_subset = list()
        for subset_idx in range(self.n_subsets):
            encoder_name_str = 'encoder_{}'.format(str(subset_idx + 1).zfill(3))
            feats_i = self.encoders[encoder_name_str](x)
            feats_per_subset.append(feats_i)

        #########################################################

        feats_per_subset_unsqueezed = [feats_i.unsqueeze(1) for feats_i in feats_per_subset]
        # concatenate all subsets, in subset dimension
        out = torch.cat(feats_per_subset_unsqueezed, dim=1)  # B x subsets x C [ x ...]

        #########################################################

        out_dict = {}
        out_dict['x'] = out

        return out_dict

class Classifier_Shared(nn.Module):
    def __init__(self,
                 ch_out,
                 n_classes,
                 dropout_1_prob,
                 stage_1_is_shared,
                 classifier_indices):
        super().__init__()
        self.ch_out = ch_out
        self.n_classes = n_classes
        self.stage_1_is_shared = stage_1_is_shared
        self.classifier_indices = classifier_indices

        if self.classifier_indices:
            encoder = get_backbone_eegnetv4(k_width=1,
                                            in_chans=16, # TODO: set this automatically, instead of hard-coded
                                            input_window_samples=400, # TODO: set this automatically, instead of hard-coded
                                            n_classes=2, # TODO: set this automatically, instead of hard-coded
                                            dropout_1_prob=dropout_1_prob,
                                            backbone_indices=self.classifier_indices)
            classifier = get_classifier(ch_out, self.n_classes)
            self.classifier = nn.Sequential(encoder,
                                            classifier)
        else:
            self.classifier = get_classifier(ch_out, self.n_classes)

    def forward(self, x_dict):

        if isinstance(x_dict, dict):
            x = x_dict['x']
        else:
            print('Input should be a dictionary. Quitting...')
            quit()

        if self.stage_1_is_shared:
            scores = self.classifier(x)
        else:
            scores_per_subset = []
            for subset_idx in range(x.shape[1]):
                scores_i = self.classifier(x[:,subset_idx])
                scores_per_subset.append(scores_i)
            scores_per_subset_unsqueezed = [scores_i.unsqueeze(1) for scores_i in scores_per_subset]
            scores = torch.cat(scores_per_subset_unsqueezed, dim=1)

        out_dict = {}
        out_dict['scores'] = scores

        return out_dict

class Classifier_Multi(nn.Module):
    def __init__(self,
                 n_subsets,
                 ch_out,
                 n_classes,
                 dropout_1_prob,
                 stage_1_is_shared,
                 classifier_indices):
        super().__init__()
        self.n_subsets = n_subsets
        self.ch_out = ch_out
        self.n_classes = n_classes
        self.stage_1_is_shared = stage_1_is_shared
        self.classifier_indices = classifier_indices

        self.classifiers = nn.ModuleDict({})
        for subset_idx in range(self.n_subsets):
            classifier_name_str = 'classifier_{}'.format(str(subset_idx).zfill(3))
            if self.classifier_indices:
                encoder = get_backbone_eegnetv4(k_width=1, # TODO: set this automatically, instead of hard-coded
                                                in_chans=16, # TODO: set this automatically, instead of hard-coded
                                                input_window_samples=400, # TODO: set this automatically, instead of hard-coded
                                                n_classes=2, # TODO: set this automatically, instead of hard-coded
                                                dropout_1_prob=dropout_1_prob,
                                                backbone_indices=self.classifier_indices)
                classifier = get_classifier(ch_out, self.n_classes)
                classifier_module = nn.Sequential(encoder, classifier)
            else:
                classifier_module = get_classifier(ch_out, self.n_classes)
            self.classifiers.update({classifier_name_str: classifier_module})

    def forward(self, x_dict):

        if isinstance(x_dict, dict):
            x = x_dict['x']
        else:
            print('Input should be a dictionary. Quitting...')
            quit()
        ######################################################
        scores_per_subset = []
        if self.stage_1_is_shared:
            for subset_idx in range(self.n_subsets):
                classifier_name_str = 'classifier_{}'.format(str(subset_idx).zfill(3))
                scores_i = self.classifiers[classifier_name_str](x)
                scores_per_subset.append(scores_i)
        else:
            for subset_idx in range(self.n_subsets):
                classifier_name_str = 'classifier_{}'.format(str(subset_idx).zfill(3))
                scores_i = self.classifiers[classifier_name_str](x[:,subset_idx])
                scores_per_subset.append(scores_i)
        scores_per_subset_unsqueezed = [scores_i.unsqueeze(1) for scores_i in scores_per_subset]
        scores = torch.cat(scores_per_subset_unsqueezed, dim=1) # concatenate all subsets in subset dim.
        ######################################################
        out_dict = {}
        out_dict['scores'] = scores

        return out_dict

class EnsembleEEG(nn.Module):
    def __init__(self,
                 n_datasets_train,
                 in_chans,
                 input_window_samples,
                 dropout_input_prob,
                 dropout_1_prob,
                 n_subsets,
                 k_width,
                 n_classes,
                 n_modules_backbone,
                 stage_1_is_shared,
                 stage_2_is_shared,
                 ):
        super().__init__()
        self.n_datasets_train = n_datasets_train
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.n_subsets = n_subsets
        self.k_width = k_width
        self.n_classes = n_classes
        self.module_indices = [0, 1, 2, 3]
        self.n_modules_backbone = n_modules_backbone
        self.backbone_indices = self.module_indices[:self.n_modules_backbone]
        self.classifier_indices = self.module_indices[self.n_modules_backbone:]
        self.stage_1_is_shared = stage_1_is_shared
        self.stage_2_is_shared = stage_2_is_shared

        if self.stage_1_is_shared:
            self.stage_1 = Backbone_Shared(n_subsets=n_subsets,
                                           k_width=k_width,
                                           in_chans=in_chans,
                                           input_window_samples=input_window_samples,
                                           n_classes=n_classes,
                                           dropout_input_prob=dropout_input_prob,
                                           dropout_1_prob=dropout_1_prob,
                                           backbone_indices=self.backbone_indices)
        else:
            self.stage_1 = Backbone_Multi(n_subsets=n_subsets,
                                          k_width=k_width,
                                          in_chans=in_chans,
                                          input_window_samples=input_window_samples,
                                          n_classes=n_classes,
                                          n_datasets=n_datasets_train,
                                          dropout_input_prob=dropout_input_prob,
                                          dropout_1_prob=dropout_1_prob,
                                          backbone_indices=self.backbone_indices)
        stage_1_num_features = self.stage_1.num_features

        if self.stage_2_is_shared:
            self.stage_2 = Classifier_Shared(ch_out=stage_1_num_features,
                                             n_classes=n_classes,
                                             dropout_1_prob=dropout_1_prob,
                                             stage_1_is_shared=self.stage_1_is_shared,
                                             classifier_indices=self.classifier_indices)
        else:
            self.stage_2 = Classifier_Multi(n_subsets=n_subsets,
                                            ch_out=stage_1_num_features,
                                            n_classes=n_classes,
                                            dropout_1_prob=dropout_1_prob,
                                            stage_1_is_shared=self.stage_1_is_shared,
                                            classifier_indices=self.classifier_indices)

    def forward(self, x_dict):
        if not isinstance(x_dict, dict):
            print('Input should be a dictionary. Quitting...')
            quit()
        feats_1 = self.stage_1(x_dict)
        out_dict = self.stage_2(feats_1)
        out_dict['feats'] = feats_1['x']
        return out_dict