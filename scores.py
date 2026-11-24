import numpy as np
from skimage import metrics
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy

def SNR2(input, output, target):
       
    diff_clean = output - target
    mse_pred_clean = np.mean(np.power(diff_clean, 2))

    diff_noisy = input - target
    mse_noisy_clean = np.mean(np.power(diff_noisy, 2))

    snr2 = 1 - (np.sqrt(mse_pred_clean) / np.sqrt(mse_noisy_clean))

    return snr2

def PSNR(output, target):          
    psnr = metrics.peak_signal_noise_ratio(target, output, data_range=2)
    return psnr

def MSE(output, target):
    mse = mean_squared_error(target, output)
    return mse

def SSIM(output, target):
    ssim = metrics.structural_similarity(target, output, data_range=1.0, channel_axis=-1)
    return ssim

def KL(output, target, bins: int = 256, eps: float = 1e-10):
    if target.ndim != output.ndim:
        raise ValueError("As imagens 'target' e 'output' devem ter o mesmo número de dimensões.")
    # Se for escala de cinza (2D), adiciona um eixo extra para tratar igual a colorida (HxWx1)
    if target.ndim == 2:
        target = target[:, :, np.newaxis]
        output = output[:, :, np.newaxis]

    canais = target.shape[2]
    kl_por_canal = []

    for c in range(canais):
        # Extrai o histograma de cada canal
        hist_target, _ = np.histogram(target[:, :, c], bins=bins, range=(0, 256))
        hist_output, _ = np.histogram(output[:, :, c], bins=bins, range=(0, 256))

        # Converte para distribuição de probabilidade e adiciona epsilon
        p = (hist_target.astype(np.float64) / hist_target.sum()) + eps
        q = (hist_output.astype(np.float64) / hist_output.sum()) + eps

        # Re-normaliza para garantir que a soma continue exatamente 1 após somar o eps
        p /= p.sum()
        q /= q.sum()

        # Calcula D_KL(P || Q) usando scipy.stats.entropy
        kl = entropy(p, q)
        kl_por_canal.append(kl)

    # Retorna a média da divergência KL entre os canais
    return float(np.mean(kl_por_canal))

def export_metrics_csv(input, output, results_dir, target=None):

    if target is None:
        return None

    ssim = SSIM(output, target)
    kl = KL(output, target)

    metrics = np.array([
        ssim,
        kl
    ])

    headers = [
        "ssim",
        "kl"
    ]

    np.savetxt(
        f'{results_dir}/metrics.csv',
        metrics.reshape(1, -1),
        delimiter=",",
        fmt="%.4f",
        header=",".join(headers),
        comments=""
    )

    return f'ssim: {ssim:.4f}, kl: {kl:.4f}'