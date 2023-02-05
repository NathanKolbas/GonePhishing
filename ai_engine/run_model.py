import pickle

import pandas as pd
import torch
import numpy as np

from model_training import MLP

PATH = "trained_model.pt"


def run_model(url):
    # Clean up the data
    data = [80, 13, 1, 3, 0, 0, 1, 1, 0, 2, 2, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.325, 0.0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 9, 4, 2, 3, 2, 16, 5, 16, 6.555555555555555, 4.0, 7.285714285714286, 0, 0, 0, 0, 0, 0]

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    X_scaled = scaler.transform([data])
    print(X_scaled)

    new_X = pd.DataFrame(data=X_scaled)
    new_X = np.array(new_X)
    train_input_tensor = torch.from_numpy(new_X).float().cuda()

    print(train_input_tensor)

    loaded_model = torch.load(PATH, map_location=torch.device('cuda'))
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(train_input_tensor)
        print(f'outputs: {outputs}')
        print(f'Confidence: {outputs.item()}')
        result = torch.round(outputs)
        print(f'Rounded: {result.item()}')
        return result.item()


if __name__ == '__main__':
    run_model('')
