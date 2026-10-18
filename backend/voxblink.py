"""Modelo de voz SimAM-ResNet34 entrenado con VoxBlink2 (100.000+ voces de YouTube en muchos idiomas)
y afinado con VoxCeleb2. Pesos: huggingface.co/gaunernst/wespeaker-voxblink2-samresnet34-ft
(formato WeSpeaker; la librería no se puede instalar aquí, así que la red se reconstruye tal cual).

Entrada: audio 16 kHz. Salida: huella de 256 números.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

RAIZ = Path(__file__).resolve().parent.parent
PESOS = {"r34": RAIZ / "modelos" / "voxblink2" / "avg_model.pt",          # SimAM-ResNet34
         "r100": RAIZ / "modelos" / "voxblink2-r100" / "avg_model.pt"}    # SimAM-ResNet100 (bzikst/voxblink2_samresnet100_ft)
BLOQUES = {"r34": (3, 4, 6, 3), "r100": (6, 16, 24, 3)}


class BloqueSimAM(nn.Module):
    def __init__(self, entrada, salida, paso=1):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, salida, 3, paso, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(salida)
        self.conv2 = nn.Conv2d(salida, salida, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(salida)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential()
        if paso != 1 or entrada != salida:
            self.downsample = nn.Sequential(nn.Conv2d(entrada, salida, 1, paso, bias=False), nn.BatchNorm2d(salida))

    @staticmethod
    def simam(x, lam=1e-4):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        return x * torch.sigmoid(d / (4 * (v + lam)) + 0.5)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.simam(self.bn2(self.conv2(out)))
        return self.relu(out + self.downsample(x))


class ResNet(nn.Module):
    def __init__(self, planos=64, bloques=(3, 4, 6, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, planos, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planos)
        self.relu = nn.ReLU()
        capas, entrada = [], planos
        for i, n in enumerate(bloques):
            salida = planos * 2 ** i
            capa = [BloqueSimAM(entrada, salida, 1 if i == 0 else 2)] + [BloqueSimAM(salida, salida) for _ in range(n - 1)]
            capas.append(nn.Sequential(*capa))
            entrada = salida
        self.layer1, self.layer2, self.layer3, self.layer4 = capas

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ASP(nn.Module):
    """Attentive statistics pooling."""
    def __init__(self, canales):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(canales, 128, 1), nn.ReLU(), nn.BatchNorm1d(128), nn.Conv1d(128, canales, 1), nn.Softmax(dim=2))

    def forward(self, x):
        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum(x ** 2 * w, dim=2) - mu ** 2).clamp(min=1e-5))
        return torch.cat((mu, sg), 1)


class SimAMResNet34(nn.Module):
    def __init__(self, mel=80, dim=256, bloques=(3, 4, 6, 3)):
        super().__init__()
        self.front = ResNet(bloques=bloques)
        self.pooling = ASP(512 * mel // 8)
        self.bottleneck = nn.Linear(512 * mel // 8 * 2, dim)

    def forward(self, fbank):                  # (lote, tramas, 80)
        x = fbank.permute(0, 2, 1).unsqueeze(1)  # (lote, 1, 80, tramas)
        return self.bottleneck(self.pooling(self.front(x)))


@lru_cache(maxsize=2)
def modelo(variante: str = "r34"):
    m = SimAMResNet34(bloques=BLOQUES[variante])
    m.load_state_dict(torch.load(PESOS[variante], map_location="cpu", weights_only=False))
    return m.eval()


def fbank(audio: np.ndarray) -> torch.Tensor:
    """Igual que WeSpeaker: fbank de Kaldi (80 mel, 25/10 ms, Hamming) y media restada."""
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None] * (1 << 15)
    f = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                    sample_frequency=16000, window_type="hamming", use_energy=False)
    return f - f.mean(0)


@torch.no_grad()
def huella(audio: np.ndarray, variante: str = "r34") -> np.ndarray:
    return modelo(variante)(fbank(audio)[None])[0].numpy()


def huellas(audio: np.ndarray, variante: str = "r34") -> np.ndarray:
    """Una huella normalizada por cada trozo de 6 s (como voz.huellas)."""
    from voz import trozos
    partes = trozos(audio)
    if not partes:
        return np.zeros((0, 256), dtype=np.float32)
    v = np.stack([huella(p, variante) for p in partes]).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
