"""
Main file for launch application
"""
import numpy as np

from icscreator.kernel.prepared_models import VMLSTMModel
from icscreator.kernel.visualization import VisualTool

DATA = "../static/data/Data_JC.csv"
MODEL_PATH = "../static/models"

data = np.genfromtxt(DATA, delimiter=',', skip_header=True)
fuel = np.expand_dims(data[::100, 1], axis=-1) / 4.0
freq = np.expand_dims(data[::100, 2], axis=-1) / 200000.0
temp = np.expand_dims(data[::100, 3], axis=-1) / 1000.0
output = np.concatenate([freq, temp], axis=-1)

network_model = VMLSTMModel((1,), (5, 5, 5), (2,))
network_model.compile('mse', 'adam', {'learning_rate': 0.00001})
network_model.fit(fuel, output, epochs=100, model_path=MODEL_PATH)

vis_tool = VisualTool(
    titles=['Rotor frequency', 'Turbine temperature'],
    x_info=['Time step'] * 2,
    y_info=['Normalized frequency', 'Normalized temperature'],
    legend=['Model', 'Target'],
    ratio=1 / 2,
    show_loss=False
)
predicted_data = network_model.predict(fuel)
vis_tool.draw([predicted_data, output])
vis_tool.show()
