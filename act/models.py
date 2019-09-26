import torch
from torch import nn


class SRNCell(nn.Module):
    def __init__(self, cfg, controller):
        super(SRNCell, self).__init__()
        self.cfg = cfg
        self.controller = controller
        self.hidden2ans = nn.Linear(cfg.MODEL.CONTROLLER.HIDDEN_SIZE,
                                    cfg.OUTPUT.DIM)

    def forward(self, x, previous_state, adaptive=True):
        _, previous_hidden, previous_ponder_cost, previous_steps = previous_state

        batch_size = previous_ponder_cost.shape[0]
        
        # s_n when n=1 is s_{t-1}
        s_n = previous_hidden

        # for statistics purposes, contains the number of steps before halting
        steps = previous_steps + 1

        # binary flag for x_n
        b_flag = x.new_ones(
            (batch_size, 1)) if self.cfg.MODEL.MAX_ITER == 1 else x.new_zeros((batch_size, 1))
        x_n = torch.cat((x, b_flag), 1)

        # run controller using x_n and sub-state (s_n) of previous iteration
        s_n = self.controller(x_n, s_n)
        if self.cfg.MODEL.LSTM:
            controller_output = s_n[0]
        else:
            controller_output = s_n

        # use new sub-state
        y_n = self.hidden2ans(controller_output)

        # accumulate s and y for timestep t
        y = y_n
        s = s_n

        ponder_cost = previous_ponder_cost

        return y, s, ponder_cost, steps


class ACTCell(nn.Module):
    def __init__(self, cfg, controller):
        super(ACTCell, self).__init__()
        self.cfg = cfg
        self.controller = controller
        self.hidden2halt = nn.Linear(cfg.MODEL.CONTROLLER.HIDDEN_SIZE, 1)
        self.hidden2ans = nn.Linear(cfg.MODEL.CONTROLLER.HIDDEN_SIZE,
                                    cfg.OUTPUT.DIM)

    def forward(self, x, previous_state, adaptive=True):
        _, previous_hidden, previous_ponder_cost, previous_steps = previous_state

        batch_size = previous_ponder_cost.shape[0]

        # s accumulates p_n * s_n
        if self.cfg.MODEL.LSTM:
            s = (torch.zeros_like(previous_hidden[0]), torch.zeros_like(previous_hidden[1]))
        else:
            s = torch.zeros_like(previous_hidden)
        # y accumulates p_n * y_n
        y = x.new_zeros((batch_size, self.cfg.OUTPUT.DIM))
        # h accumulates h_n
        h = x.new_zeros((batch_size, 1))

        # s_n when n=1 is s_{t-1}
        s_n = previous_hidden
        # remainder for each step
        r_n = x.new_ones((batch_size, 1))

        # like previous ponder cost ?
        R = x.new_ones((batch_size, 1))
        N = x.new_zeros((batch_size, 1))

        # TODO: register as buffer
        ε = x.new_full((1,), self.cfg.ACT.BASELINE.EPSILON)

        # for statistics purposes, contains the number of steps before halting
        steps = previous_steps + 1

        for n in range(1, self.cfg.MODEL.MAX_ITER):

            # binary flag for x_n
            b_flag = x.new_ones(
                (batch_size, 1)) if n == 1 else x.new_zeros((batch_size, 1))
            x_n = torch.cat((x, b_flag), 1)

            # run controller using x_n and sub-state (s_n) of previous iteration
            s_n = self.controller(x_n, s_n)
            if self.cfg.MODEL.LSTM:
                controller_output = s_n[0]
            else:
                controller_output = s_n

            # use new sub-state
            h_n = torch.sigmoid(self.hidden2halt(controller_output))
            if self.cfg.OUTPUT.DIM == 1:
                y_n = self.hidden2ans(controller_output)
            elif self.cfg.MODEL.TASK == "addition":
                y_n = self.hidden2ans(controller_output)
            else:
                y_n = self.hidden2ans(controller_output)

            # check if n >= N(t) using ∑h_n
            h += h_n
            isN = (h >= (x.new_ones(1) - ε))

            # compute p_n
            p_n = torch.where(isN, r_n, h_n)

            # accumulate s and y for timestep t
            y += p_n * y_n
            if self.cfg.MODEL.LSTM:
                s = (torch.add(p_n * s_n[0], s[0]), torch.add(p_n * s_n[1], s[1]))
            else:
                s += p_n * s_n

            # break loop if none in batch can change ans (ACT)
            if isN.all():  # and adaptive
                current_ponder_cost = (R + N)
                ponder_cost = previous_ponder_cost + current_ponder_cost
                return y, s, ponder_cost, steps

            steps = torch.where(isN, steps, steps + 1)

            # remainder for next timestep
            r_n = torch.where(isN, x.new_zeros(1), x.new_ones(1) - h)

            # update values of R and N
            # while not n=N(t) -> R = r_{n+1}
            R = torch.where(isN, R, r_n)
            # while not n=N(t) -> N = n+1
            N = torch.where(isN, N, x.new_full((1,), n + 1))

        # for last iteration p_n=r_n so as to follow valid prob. distribution

        # binary flag for x_n
        b_flag = x.new_ones(
            (batch_size, 1)) if self.cfg.MODEL.MAX_ITER == 1 else x.new_zeros((batch_size, 1))
        x_n = torch.cat((x, b_flag), 1)

        # run controller using x_n and sub-state (s_n) of previous iteration
        s_n = self.controller(x_n, s_n)
        if self.cfg.MODEL.LSTM:
            controller_output = s_n[0]
        else:
            controller_output = s_n

        # use new sub-state
        if self.cfg.OUTPUT.DIM == 1:
            y_n = self.hidden2ans(controller_output)
        elif self.cfg.MODEL.TASK == "addition":
            y_n = self.hidden2ans(controller_output)
        else:
            y_n = self.hidden2ans(controller_output)

        # compute p_n
        p_n = r_n

        # accumulate s and y for timestep t
        y += p_n * y_n
        if self.cfg.MODEL.LSTM:
            s = (torch.add(p_n * s_n[0], s[0]),
                torch.add(p_n * s_n[1], s[1]))
        else:
            s += p_n * s_n

        current_ponder_cost = (R + N)
        ponder_cost = previous_ponder_cost + current_ponder_cost
        return y, s, ponder_cost, steps


class BaseACT(nn.Module):
    def __init__(self, cfg):
        super(BaseACT, self).__init__()
        self.cfg = cfg

        if cfg.MODEL.GRU:
            controller = nn.GRUCell(cfg.INPUT.DIM + 1,
                                    cfg.MODEL.CONTROLLER.HIDDEN_SIZE)
        elif cfg.MODEL.LSTM:
            controller = nn.LSTMCell(cfg.INPUT.DIM + 1,
                                    cfg.MODEL.CONTROLLER.HIDDEN_SIZE)
        else:
            controller = nn.RNNCell(cfg.INPUT.DIM + 1,
                                    cfg.MODEL.CONTROLLER.HIDDEN_SIZE)

        if cfg.MODEL.RNN_BASELINE:
            self.cell = SRNCell(cfg, controller)
        else:
            self.cell = ACTCell(cfg, controller)

    def init_hidden(self, device):
        if self.cfg.MODEL.LSTM:
            hx = torch.zeros(self.cfg.DATALOADER.BATCH_SIZE,
                                   self.cfg.MODEL.CONTROLLER.HIDDEN_SIZE
                                   ).to(device)
            cx = torch.zeros_like(hx)
            hidden_state = (hx, cx)
        else:
            hidden_state = torch.zeros(self.cfg.DATALOADER.BATCH_SIZE,
                                   self.cfg.MODEL.CONTROLLER.HIDDEN_SIZE
                                   ).to(device)
        ponder_cost = torch.full((self.cfg.DATALOADER.BATCH_SIZE, 1), self.cfg.ACT.MIN_PENALTY).to(device)
        steps = torch.zeros(self.cfg.DATALOADER.BATCH_SIZE, 1).to(device)

        return None, hidden_state, ponder_cost, steps

    def forward(self, input, hidden, adaptive=False):
        output = []
        for i in range(input.size(0)):
            hidden = self.cell(input[i], hidden, adaptive)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden
