"""MLP, BD-LSTM, Conv-LSTM (PyTorch). Đầu vào (B, W, F) -> đầu ra (B, H) (direct multi-output)."""
from __future__ import annotations
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features, window, horizon, hidden=20):
        super().__init__()
        d = n_features * window
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))  # 3 hidden layers

    def forward(self, x):
        return self.net(x)


class BDLSTM(nn.Module):
    """Hai LSTM chạy xuôi/ngược trên cửa sổ, nối trạng thái cuối rồi Dense."""
    def __init__(self, n_features, window, horizon, hidden=100):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def forward(self, x):
        _, (h, _) = self.lstm(x)                 # h: (2, B, hidden)
        return self.head(torch.cat([h[0], h[1]], dim=1))


class ConvLSTM1DCell(nn.Module):
    """LSTM cell: thay phép nhân ma trận ở các cổng bằng tích chập 1D."""
    def __init__(self, in_ch, hid_ch, kernel=2):
        super().__init__()
        self.hid = hid_ch
        self.conv = nn.Conv1d(in_ch + hid_ch, 4 * hid_ch, kernel, padding="same")

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Cửa sổ W chia thành S=W/sub_len đoạn con (thời gian) x sub_len bước (không gian), kênh = số đặc trưng
    (cách dùng ConvLSTM1D phổ biến cho chuỗi thời gian). Tích chập bắt mẫu cục bộ, LSTM bắt phụ thuộc thời gian."""
    def __init__(self, n_features, window, horizon, hidden=20, sub_len=3, kernel=2):
        super().__init__()
        assert window % sub_len == 0, "window phải chia hết cho sub_len"
        self.S, self.L, self.hid = window // sub_len, sub_len, hidden
        self.cell = ConvLSTM1DCell(n_features, hidden, kernel)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden * sub_len, horizon))

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, self.S, self.L, -1).permute(0, 1, 3, 2)   # (B, S, F, L)
        h = x.new_zeros(B, self.hid, self.L)
        c = x.new_zeros(B, self.hid, self.L)
        for s in range(self.S):
            h, c = self.cell(x[:, s], (h, c))
        return self.head(h)


def build_model(name, n_features, window, horizon, hidden):
    cls = {"MLP": MLP, "BD-LSTM": BDLSTM, "Conv-LSTM": ConvLSTM}[name]
    return cls(n_features, window, horizon, hidden)


def artifact_name(coin, strategy, model):
    return f"{coin}_{strategy}_{model}.pt"
