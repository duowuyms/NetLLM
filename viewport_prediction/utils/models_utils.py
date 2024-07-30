from models.regression import create_linear_regression
from models.velocity import create_velocity
from models.track import create_track


def create_model(model_name, his_window, fut_window, device, seed):
    model = None
    if model_name == 'regression':
        model = create_linear_regression(his_window, fut_window, device, seed)
    elif model_name == 'velocity':
        model = create_velocity(fut_window=fut_window)
    elif model_name == 'track':
        model = create_track(his_window, fut_window, device, HEIGHT=224, WIDTH=224)
    return model
