import numpy as np
import re
import os
import matplotlib.pyplot as plt


def vmin_vmax_percentile(target):
    vmin = np.percentile(target, 2)
    vmax = np.percentile(target, 98)

    if np.abs(vmin) > np.abs(vmax):
        vmax = np.abs(vmin)
    else:
        vmin = -np.abs(vmax)
    
    return vmin, vmax


def plot_wavelet(
    wavelet: np.ndarray,
    results_dir: str,
    dt: float = None,
    true_wavelet: np.ndarray = None,
    title: str = "Wavelet",
    ax: plt.Axes = None,
):
    """
    Plota a wavelet predita (amplitude x tempo), no mesmo estilo dos
    gráficos de wavelet do artigo (ex.: Figuras 6, 12, 17).
 
    Parâmetros
    ----------
    wavelet : np.ndarray ou torch.Tensor, shape (L,)
        Wavelet predita (por exemplo, `resultado['wavelet']` retornado
        por `train_ssl_bd`, ou `outputs['wavelet']` do forward do
        modelo).
    dt : float, opcional
        Intervalo de amostragem no tempo (em segundos). Se fornecido, o
        eixo x é mostrado em segundos, centralizado em torno de zero
        (como nas Figuras 6, 12 e 17). Se None, o eixo x mostra apenas o
        índice da amostra.
    true_wavelet : np.ndarray, opcional
        Wavelet verdadeira (ground truth), se disponível, para sobrepor
        a curva estimada com a curva real -- reproduzindo a comparação
        feita na Figura 6 do artigo ("Estimated" vs "True").
    title : str
        Título do gráfico.
    ax : matplotlib.axes.Axes, opcional
        Eixo onde plotar. Se None, cria uma nova figura.
 
    Retorna
    -------
    matplotlib.axes.Axes
        O eixo com o gráfico plotado.
    """
    # aceita tanto np.ndarray quanto torch.Tensor sem exigir import de torch
    wavelet = np.asarray(
        wavelet.detach().cpu().numpy() if hasattr(wavelet, "detach") else wavelet,
        dtype=float,
    )
 
    n = wavelet.shape[0]
 
    if dt is not None:
        # eixo do tempo centralizado em zero, como nas Figuras 6/12/17
        t = (np.arange(n) - n // 2) * dt
        xlabel = "Time (s)"
    else:
        t = np.arange(n)
        xlabel = "Sample index"
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
 
    if true_wavelet is not None:
        true_wavelet = np.asarray(
            true_wavelet.detach().cpu().numpy() if hasattr(true_wavelet, "detach") else true_wavelet,
            dtype=float,
        )
        ax.plot(t, true_wavelet, color="black", linestyle="-", label="True")
        ax.plot(t, wavelet, color="red", linestyle="--", label="Estimated")
        ax.legend()
    else:
        ax.plot(t, wavelet, color="red", linestyle="-", label="Estimated")
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
 
    plt.savefig(f'{results_dir}/wavelet.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/wavelet.pdf', bbox_inches='tight', transparent=False)

    plt.close()


def plot_wavelets(directory="wavelets", cmap="Blues", title: str = "Wavelet"):
    """
    Ordena e plota todas as wavelets armazenadas no diretório.

    Os arquivos devem seguir o padrão:
        wavelet_epoch_X.npy

    Parâmetros
    ----------
    directory : str
        Diretório contendo os arquivos .npy.
    cmap : str
        Colormap do Matplotlib usado para o gradiente de cores.
    """

    # Encontrar arquivos no formato wavelet_epoch_X.npy
    pattern = re.compile(r"wavelet_epoch_(\d+)\.npy$")

    wavelet_files = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)

        if match:
            epoch = int(match.group(1))
            wavelet_files.append((epoch, filename))

    if not wavelet_files:
        raise ValueError(
            f"Nenhum arquivo 'wavelet_epoch_X.npy' encontrado em '{directory}'."
        )

    # Ordenar pela época
    wavelet_files.sort(key=lambda x: x[0])

    # Criar figura
    plt.figure(figsize=(12, 6))

    for (epoch, filename) in wavelet_files:

        filepath = os.path.join(directory, filename)
        wavelet = np.load(filepath)

        plt.plot(
            wavelet,
            linewidth=1.5,
            label=f"Epoch {epoch}"
        )

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(alpha=0.2)

    plt.legend(fontsize='xx-small')

    plt.tight_layout()
    
    plt.savefig(f'{directory}/wavelet_history.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{directory}/wavelet_history.pdf', bbox_inches='tight', transparent=False)

    plt.close()


def plot_comparison(input, output, results_dir, name, metrics=None, target=None):
    vmin, vmax = vmin_vmax_percentile(target)

    fig = plt.figure(figsize=(24,5))
    fig.subplots_adjust(bottom=0.05, left=0.05, top = 0.975, right=0.975)
    fig.tight_layout()

    # Plots
    ax = fig.add_subplot(1, 4, (1))
    ax.imshow(input, cmap="seismic", interpolation='hanning', interpolation_stage='data', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_ylabel('Time [ms]',fontsize=15)
    ax.set_xlabel('Trace number',fontsize=15)
    ax.set_title('(a)', y=-0.2,fontsize=20)
    

    ax = fig.add_subplot(1, 4, (2))
    im = ax.imshow(target, cmap="seismic", interpolation='hanning', interpolation_stage='data', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('Trace number',fontsize=15)
    ax.set_title('(b)', y=-0.2,fontsize=20)

    ax = fig.add_subplot(1, 4, (3))
    ax.imshow(output, cmap="seismic", interpolation='hanning', interpolation_stage='data', vmin=vmin, vmax=vmax, aspect='auto')
    if metrics:
        ax.text(0.99, 0.99, metrics, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)

    ax.set_xlabel('Trace number',fontsize=15)
    ax.set_title('(c)', y=-0.2,fontsize=20)
    
    ax_diff = fig.add_subplot(1, 4, (4))
    
    diff = output - target

    ax_diff.imshow(diff, cmap="seismic", interpolation='hanning', interpolation_stage='data', vmin=vmin, vmax=vmax, aspect='auto')

    ax_diff.set_xlabel('Trace number',fontsize=15)
    ax_diff.set_title('(d)', y=-0.2,fontsize=20)
    
    

    cbar = fig.colorbar(im, ax=fig.axes, pad=0.01, aspect=80)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(f'{results_dir}/{name}.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/{name}.pdf', bbox_inches='tight', transparent=False)
    plt.close()