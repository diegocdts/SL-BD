import numpy as np
import segyio


def max_min_normalization(sample):
        
        sample = np.asarray(sample, dtype=float)
        min_val = np.abs(np.min(sample))
        max_val = np.abs(np.max(sample))

        out = sample.copy()
        out = np.where(out < 0, out / min_val, out)
        out = np.where(out > 0, out / max_val, out)
        
        return out


def hessian_normalization(sample, eps = 1e-10):
        
        max_val = np.max(sample)
        min_val = eps
        
        out  = sample / max_val
        out = np.where(out < min_val, min_val, out)
        return out


def normalization_by_hessian(out, hessian, inv):
        # Normalize the sample by its Hessian values
        if inv:
            out = out / hessian
        else:
            out = out * hessian
        return out


def zScore(data, mean = None, std = None, n_std = 3):
    mean = np.mean(data) if mean is None else mean
    std = np.std(data) if mean is None else std
    return (data) / (std * n_std)


def load_data(data_path: str):
    if '.npy' in data_path:
        data = np.load(data_path).astype("float32")
    else:
        with segyio.open(data_path, ignore_geometry=False) as file:
            data = file.iline[1]
    data = data[200:532,3500:4900]
    data = data if data.ndim == 2 else data[0]
    data = data.reshape(data.shape[-2], data.shape[-1])

    return data


def norm_data(data, hess):
    hessian = hessian_normalization(hess)
    sample = max_min_normalization(data)
    hsample = normalization_by_hessian(sample, hessian, True)
    final_sample = max_min_normalization(hsample)

    return final_sample