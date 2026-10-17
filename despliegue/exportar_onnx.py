"""Exporta el modelo de huellas de voz a ONNX para calcularlas en el navegador.

La STFT de SpeechBrain no se puede exportar, así que se rehace como una convolución
con los mismos parámetros. El resto (filtros mel, normalización y red ECAPA) es el original.

Uso: .venv/bin/python despliegue/exportar_onnx.py  ->  frontend/modelo/ecapa-int8.onnx
(comprimido a 8 bits: 22 MB en vez de 84, mismo acierto)
"""
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
import voz  # noqa: E402


class HuellaVoz(torch.nn.Module):
    """Audio mono 16 kHz (1, muestras) -> huella (1, 192)."""

    def __init__(self, m):
        super().__init__()
        f = m.mods.compute_features
        stft = f.compute_STFT
        n_fft, self.hop = stft.n_fft, stft.hop_length
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * torch.pi * k * t / n_fft
        ventana = stft.window.double()
        kernel = torch.cat([torch.cos(ang) * ventana, -torch.sin(ang) * ventana])  # (2*bins, n_fft)
        self.register_buffer("kernel", kernel.float()[:, None, :])
        self.pad = n_fft // 2
        self.bins = n_fft // 2 + 1
        self.fbanks = f.compute_fbanks
        self.norm = m.mods.mean_var_norm
        self.red = m.mods.embedding_model

    def forward(self, wav):
        x = torch.nn.functional.pad(wav[:, None, :], (self.pad, self.pad))
        y = torch.nn.functional.conv1d(x, self.kernel, stride=self.hop)       # (1, 2*bins, frames)
        potencia = y[:, :self.bins] ** 2 + y[:, self.bins:] ** 2
        feats = self.fbanks(potencia.transpose(1, 2))                         # (1, frames, 80)
        feats = self.norm(feats, torch.ones(1))
        return self.red(feats).squeeze(1)


def main():
    m = voz.modelo()
    net = HuellaVoz(m).eval()
    salida = RAIZ / "frontend" / "modelo" / "ecapa.onnx"
    salida.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.randn(1, 16000 * 6) * 0.1
    torch.onnx.export(net, (wav,), str(salida), input_names=["wav"], output_names=["huella"],
                      dynamic_axes={"wav": {1: "muestras"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort
    sesion = ort.InferenceSession(str(salida))
    for seg in (3, 6, 8):
        w = torch.randn(16000 * seg) * 0.1
        a = m.encode_batch(w[None]).squeeze().detach().numpy()
        b = sesion.run(None, {"wav": w[None].numpy()})[0][0]
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        print(f"{seg} s: parecido con el modelo original = {cos:.6f}")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    final = salida.with_name("ecapa-int8.onnx")
    quantize_dynamic(str(salida), str(final), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    salida.unlink()
    print(f"Guardado {final} ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
