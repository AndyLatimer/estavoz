"""Exporta el modelo VoxBlink2 (backend/voxblink.py) a ONNX para el navegador, con el fbank de Kaldi dentro.

torchaudio.compliance.kaldi.fbank no se puede exportar tal cual, así que se rehace con operaciones
sencillas (convolución para trocear, matrices para la DFT) con los mismos pasos y parámetros:
tramas de 400 muestras cada 160, quitar la media, preénfasis 0.97, ventana de Hamming, FFT de 512,
potencia, 80 bancos mel, logaritmo y media restada (CMN).

Uso: .venv/bin/python despliegue/exportar_voxblink.py  ->  frontend/modelo/voxblink-int8.onnx
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchaudio.compliance.kaldi import get_mel_banks

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
import voxblink  # noqa: E402

N, PASO, NFFT, MEL = 400, 160, 512, 80


class FbankKaldi(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("trocear", torch.eye(N)[:, None, :])          # conv1d: (400, 1, 400)
        n = torch.arange(N, dtype=torch.float64)[:, None]
        k = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[None, :]
        ventana = torch.hamming_window(N, periodic=False, dtype=torch.float64)[:, None]
        ang = 2 * torch.pi * n * k / NFFT
        self.register_buffer("cos", (torch.cos(ang) * ventana).float())    # (400, 257)
        self.register_buffer("sin", (torch.sin(ang) * ventana).float())
        bancos, _ = get_mel_banks(MEL, NFFT, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel", torch.nn.functional.pad(bancos, (0, 1)).T.float())  # (257, 80)
        self.eps = float(torch.finfo(torch.float32).eps)

    def forward(self, wav):                                    # (1, muestras) en [-1, 1]
        x = wav[:, None, :] * 32768.0
        tramas = torch.nn.functional.conv1d(x, self.trocear, stride=PASO)[0].T   # (tramas, 400)
        tramas = tramas - tramas.mean(dim=1, keepdim=True)
        previo = torch.cat([tramas[:, :1], tramas[:, :-1]], dim=1)
        tramas = tramas - 0.97 * previo
        re, im = tramas @ self.cos, tramas @ self.sin
        potencia = re * re + im * im                           # (tramas, 257)
        f = torch.log(torch.clamp(potencia @ self.mel, min=self.eps))
        return f - f.mean(dim=0, keepdim=True)


class VozNavegador(nn.Module):
    def __init__(self):
        super().__init__()
        self.fbank = FbankKaldi()
        self.red = voxblink.modelo()

    def forward(self, wav):
        return self.red(self.fbank(wav)[None])


def main():
    net = VozNavegador().eval()
    wav = torch.randn(1, 16000 * 6) * 0.1
    with torch.no_grad():
        a = voxblink.fbank(wav[0].numpy())
        b = net.fbank(wav)
    print(f"fbank: diferencia máxima con Kaldi {float((a - b).abs().max()):.2e}")

    salida = RAIZ / "frontend" / "modelo" / "voxblink.onnx"
    torch.onnx.export(net, (wav,), str(salida), input_names=["wav"], output_names=["huella"],
                      dynamic_axes={"wav": {1: "muestras"}}, opset_version=17, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort
    final = salida.with_name("voxblink-int8.onnx")
    # el fbank necesita precisión completa: comprimirlo a 8 bits destroza la huella (parecido 0,2)
    import onnx
    fbank = [n.name for n in onnx.load(str(salida)).graph.node if "fbank" in n.name.lower()]
    quantize_dynamic(str(salida), str(final), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=fbank)
    for ruta in (salida, final):
        s = ort.InferenceSession(str(ruta))
        cs = []
        for seg in (3, 6, 10):
            w = torch.randn(16000 * seg) * 0.1
            ref = voxblink.huella(w.numpy())
            h = s.run(None, {"wav": w[None].numpy()})[0][0]
            cs.append(float(ref @ h / np.linalg.norm(ref) / np.linalg.norm(h)))
        print(f"{ruta.name} ({ruta.stat().st_size / 1e6:.0f} MB): parecido con el original {min(cs):.4f}")
    salida.unlink()


if __name__ == "__main__":
    main()
