
from pathlib import Path
from .affine_calibration import AffineCalibrator, predict

import pandas as pd
import torch

def main(
    checkpoint_path: str,
    method: str,
    predict_logits: str,
    predict_labels: str,
    output_dir: str = 'output',
):
    # Load logits
    predict_logits = torch.log_softmax(torch.from_numpy(pd.read_csv(predict_logits, index_col=0, header=None).values).float(), dim=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)

    # Load model
    model = AffineCalibrator(method=method, num_classes=predict_logits.shape[1])
    state = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state['model'])

    # Predict
    cal_logits = predict(model, predict_logits)

    # Save
    output_dir = Path(output_dir)
    pd.DataFrame(cal_logits, index=df_predict_labels.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)

    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)