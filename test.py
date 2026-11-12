import os
import numpy as np
import torch
from pathlib import Path
from model import SSLBD
from visualization import plot_comparison2, load_data, extract_patches, reconstruct_patches, crop_data
from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from scores import export_metrics_csv


def test_ssl_bd(
    seismic_data: np.ndarray,
    ground_truth: np.ndarray = None,
    base_channels: int = 16,
    device: str = None,
    results_dir: str = None,
    test_dir: str = None
):
    """
    Executa o teste completo do SSL-BD sobre um dado sísmico 2D.

    Parâmetros
    ----------
    seismic_data : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico observado (Y_obs).
    ground_truth : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico verdadeiro (R_real).
    base_channels : int
        Número de canais da primeira camada da rede de refletividade.
    device : str, opcional
        'cuda' ou 'cpu'. Se None, usa GPU se disponível.
    results_dir: str
        Caminho do modelo salvo
    test_dir: str
            Caminho para salvar teste
    """

    assert results_dir is not None, "O path para os resultados precisa ser definido"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Passo 1 (Seção 2.2): estimativa do wavelet inicial de fase zero,
    # calculada uma única vez
    # --------------------------------------------------------------

    w0_np = estimate_zero_phase_wavelet(seismic_data)
    w0 = torch.tensor(w0_np, dtype=torch.float32, device=device)

    y_start, x_start = 200, 3500
    # dado sísmico como tensor (batch=1, canal=1, n_traces, n_samples)
    # y_patches, positions = extract_patches(seismic_data)
    # y_patch, positions = extract_patches(seismic_data, y_start, x_start, to_norm=True)
    y_obs = torch.tensor(seismic_data, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(0).unsqueeze(0)

    # ground truth como tensor (batch=1, canal=1, n_traces, n_samples)
    if ground_truth is not None:
        # x_patches, _ = extract_patches(ground_truth)
        # x_patch, _ = extract_patches(ground_truth, y_start, x_start, to_norm=True)
        r_real = torch.tensor(ground_truth, dtype=torch.float32, device=device)
        r_real = r_real.unsqueeze(0).unsqueeze(0)
    else:
        r_real = None

    # --------------------------------------------------------------
    # Passo 2: Carrega o checkpoint e o modelo.
    # --------------------------------------------------------------

    checkpoint = torch.load(
        f'{results_dir}/model.pth',
        map_location=device
    )

    model = SSLBD(w0=w0, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------------
    # Passo 3: Inferência
    # --------------------------------------------------------------

    mu = relative_sparsity_mu(1)
    with torch.no_grad():
        outputs = model(y_obs, mu=mu)
        prediction = outputs["reflectivity"].detach().cpu().numpy().squeeze()

    np.save(f'{test_dir}/reflectivity.npy', prediction)

    snr2 = export_metrics_csv(input=seismic_data, output=prediction, results_dir=test_dir, target=ground_truth)

    plot_comparison2(input=seismic_data, 
                        output=prediction, 
                        results_dir=test_dir, 
                        name='reflectivity', 
                        snr2=snr2, 
                        target=ground_truth) 
    
    return prediction


def call_test(base_channels, 
              result_dir, 
              test_y_path, 
              test_x_path = None, 
              mean_y = None, 
              std_y = None, 
              mean_x = None, 
              std_x = None, 
              reversed_polarity = False):
    test_dir = f'{result_dir}/test_{Path(test_y_path).stem}' if reversed_polarity is False else f'{result_dir}/test_{Path(test_y_path).stem}_RP'
    os.makedirs(test_dir, exist_ok=True)

    print(f'TESTING')

    seismic_data, _, _ = load_data(data_path=test_y_path, to_norm=True, mean=mean_y, std=std_y)
    print(f'Y: Shape: {seismic_data.shape}  - min: {seismic_data.min():.4f}    - max: {seismic_data.max():.4f}  - Path: {test_y_path}')

    if test_x_path is not None:
        ground_truth, _, _ = load_data(data_path=test_x_path, to_norm=True, mean=mean_x, std=std_x)
        print(f'X: Shape: {ground_truth.shape}  - min: {ground_truth.min():.4f}    - max: {ground_truth.max():.4f}   - Path: {test_x_path}')
    else:
        ground_truth = None

    if reversed_polarity:
        seismic_data = -seismic_data
        ground_truth = -ground_truth

    test_ssl_bd(seismic_data=seismic_data,
                ground_truth=ground_truth, 
                base_channels=base_channels, 
                results_dir=result_dir, 
                test_dir=test_dir)    

    print('Fim do teste')