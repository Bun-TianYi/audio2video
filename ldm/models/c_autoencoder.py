__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 18:05:21"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
# from ldm.modules.diffusionmodules.model import Encoder, Decoder
from einops import rearrange


class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):
    """ The pre-process layer for MNIST image

    """

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class C_Encoder(nn.Module):
    def __init__(self, in_channels, layer_num=2):
        super(C_Encoder, self).__init__()
        self.embeds = nn.ModuleList()
        # 对数据进行一次embed
        for i in range(layer_num):
            layer = nn.Sequential(
                nn.Conv1d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1),
                    nn.GroupNorm(2, in_channels),
                    nn.LeakyReLU())
            self.embeds.append(layer)

        # 对数据进行一次降采样
        self.down_sample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for embed in self.embeds:
            x = embed(x)
        x = self.down_sample(x)
        return x

class TD_VAE(pl.LightningModule):
    """ The full TD_VAE model with jumpy prediction.

    First, let's first go through some definitions which would help
    understanding what is going on in the following code.

    Belief: As the model is feed a sequence of observations, x_t, the
      model updates its belief state, b_t, through a LSTM network. It
      is a deterministic function of x_t. We call b_t the belief at
      time t instead of belief state, becuase we call the hidden state z
      state.

    State: The latent state variable, z.

    Observation: The observated variable, x. In this case, it represents
      binarized MNIST images

    """

    def __init__(self,
                 input_size,
                 processed_x_size,
                 belief_state_size,
                 state_size,
                 batch_key=None,
                 condition_key=None,
                 monitor=None,
                 ckpt_path=None,
                 ):
        super(TD_VAE, self).__init__()
        self.c_encode = C_Encoder(1024)
        self.batch_key = batch_key
        self.condition_key = condition_key
        self.x_size = input_size
        self.processed_x_size = processed_x_size
        self.b_size = belief_state_size
        self.z_size = state_size
        if monitor is not None:
            self.monitor = monitor
        self.ckpt_path = ckpt_path
        ## input pre-process layer
        self.process_x = PreProcess(self.x_size, self.processed_x_size)

        ## one layer LSTM for aggregating belief states
        ## One layer LSTM is used here and I am not sure how many layers
        ## are used in the original paper from the paper.
        self.lstm = nn.LSTM(input_size=self.processed_x_size,
                            hidden_size=self.b_size,
                            batch_first=True)
        self.c_lstm = nn.LSTM(input_size=1024,
                              hidden_size=self.b_size,
                              batch_first=True)

        ## Two layer state model is used. Sampling is done by sampling
        ## higher layer first.
        ## belief to state (b to z)
        ## (this is corresponding to P_B distribution in the reference;
        ## weights are shared across time but not across layers.)
        self.l2_b_to_z = DBlock(self.b_size, 50, self.z_size)  # layer 2
        self.l1_b_to_z = DBlock(self.b_size + self.z_size, 50, self.z_size)  # layer 1

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.l2_infer_z = DBlock(self.b_size + 2 * self.z_size, 50, self.z_size)  # layer 2
        self.l1_infer_z = DBlock(self.b_size + 2 * self.z_size + self.z_size, 50, self.z_size)  # layer 1

        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.l2_transition_z = DBlock(2 * self.z_size, 50, self.z_size)
        self.l1_transition_z = DBlock(2 * self.z_size + self.z_size, 50, self.z_size)

        ## state to observation
        self.z_to_x = Decoder(2 * self.z_size, 200, self.x_size)

    def forward(self, lms, condition):
        self.batch_size = lms.size()[0]
        self.x = lms
        ## pre-precess image x
        self.processed_x = self.process_x(self.x)

        ## aggregate the belief b
        self.b, (h_n, c_n) = self.lstm(self.processed_x)
        self.c_b, (_, _) = self.c_lstm(condition)

        pp = 0

    def calculate_loss(self, t1, t2):
        """ Calculate the jumpy VD-VAE loss, which is corresponding to
        the equation (6) and equation (8) in the reference.

        """

        ## Because the loss is based on variational inference, we need to
        ## draw samples from the variational distribution in order to estimate
        ## the loss function.

        ## sample a state at time t2 (see the reparametralization trick is used)
        ## z in layer 2
        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)  # 返回一个与均值张量相同尺寸的矩阵，元素为由标准正态分布采样出来的值
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma) * t2_l2_z_epsilon

        ## z in layer 1
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t2, :], t2_l2_z), dim=-1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma) * t2_l1_z_epsilon

        ## concatenate z from layer 1 and layer 2
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim=-1)

        ## sample a state at time t1
        ## infer state at time t1 based on states at time t2
        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[:, t1, :], t2_z), dim=-1))
        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + torch.exp(t1_l2_qs_z_logsigma) * t1_l2_qs_z_epsilon

        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(
            torch.cat((self.b[:, t1, :], t2_z, t1_l2_qs_z), dim=-1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma) * t1_l1_qs_z_epsilon

        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z), dim=-1)

        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[:, t1, :])
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t1, :], t1_l2_qs_z), dim=-1))

        ## state distribution at time t2 based on states at time t1 and state transition
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(t1_qs_z)
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z), dim=-1))

        ## observation distribution at time t2 based on state at time t2
        t2_x_prob = self.z_to_x(t2_z)

        #### start calculating the loss

        #### KL divergence between z distribution at time t1 based on variational distribution
        #### (inference model) and z distribution at time t1 based on belief.
        #### This divergence is between two normal distributions and it can be calculated analytically

        ## KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        loss = 0.5 * torch.sum(((t1_l2_pb_z_mu - t1_l2_qs_z) / torch.exp(t1_l2_pb_z_logsigma)) ** 2, -1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - torch.sum(t1_l2_qs_z_logsigma, -1)

        ## KL divergence between t1_l1_pb_z and t1_l1_qs_z
        loss += 0.5 * torch.sum(((t1_l1_pb_z_mu - t1_l1_qs_z) / torch.exp(t1_l1_pb_z_logsigma)) ** 2, -1) + \
                torch.sum(t1_l1_pb_z_logsigma, -1) - torch.sum(t1_l1_qs_z_logsigma, -1)

        #### The following four terms estimate the KL divergence between the z distribution at time t2
        #### based on variational distribution (inference model) and z distribution at time t2 based on transition.
        #### In contrast with the above KL divergence for z distribution at time t1, this KL divergence
        #### can not be calculated analytically because the transition distribution depends on z_t1, which is sampled
        #### after z_t2. Therefore, the KL divergence is estimated using samples

        ## state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5 * t2_l2_z_epsilon ** 2 - 0.5 * t2_l2_z_epsilon.new_tensor(2 * np.pi) - t2_l2_z_logsigma,
                          dim=-1)
        loss += torch.sum(-0.5 * t2_l1_z_epsilon ** 2 - 0.5 * t2_l1_z_epsilon.new_tensor(2 * np.pi) - t2_l1_z_logsigma,
                          dim=-1)

        ## state log probabilty at time t2 based on transition
        loss += torch.sum(
            0.5 * ((t2_l2_z - t2_l2_t_z_mu) / torch.exp(t2_l2_t_z_logsigma)) ** 2 + 0.5 * t2_l2_z.new_tensor(
                2 * np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(
            0.5 * ((t2_l1_z - t2_l1_t_z_mu) / torch.exp(t2_l1_t_z_logsigma)) ** 2 + 0.5 * t2_l1_z.new_tensor(
                2 * np.pi) + t2_l1_t_z_logsigma, -1)

        ## observation prob at time t2
        loss += -torch.sum(self.x[:, t2, :] * torch.log(t2_x_prob) + (1 - self.x[:, t2, :]) * torch.log(1 - t2_x_prob),
                           -1)
        loss = torch.mean(loss)

        return loss

    def rollout(self, images, t1, t2):
        self.forward(images)

        ## at time t1-1, we sample a state z based on belief at time t1-1
        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[:, t1 - 1, :])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma) * l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t1 - 1, :], l2_z), dim=-1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma) * l1_z_epsilon
        current_z = torch.cat((l1_z, l2_z), dim=-1)

        rollout_x = []

        for k in range(t2 - t1 + 1):
            ## predicting states after time t1 using state transition
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(current_z)
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma) * next_l2_z_epsilon

            next_l1_z_mu, next_l1_z_logsigma = self.l1_transition_z(
                torch.cat((current_z, next_l2_z), dim=-1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)
            next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma) * next_l1_z_epsilon
            next_z = torch.cat((next_l1_z, next_l2_z), dim=-1)

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim=1)

        return rollout_x

    def get_input(self, batch):
        x = batch[self.batch_key]
        x = rearrange(x, 'B T P C -> B T (P C)')    # p代表人脸landmark的特征点（point）
        x_c = batch[self.condition_key]
        x_c = rearrange(x_c, "B T C -> B C T")
        x_c = self.c_encode(x_c)
        x_c = rearrange(x_c, "B C T -> B T C")
        return x, x_c

    def training_step(self, batch, batch_idx):      # ,optimizer_idx): 这个optimizer是个啥没搞懂，全程没看见它从哪进的
        lms, hubert = self.get_input(batch)
        self(lms, hubert)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1, 2, 3, 4])
        loss = self.calculate_loss(t_1, t_2)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.parameters(),
                                  lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        # return [opt_ae, opt_disc], []
        return [opt_ae], []


