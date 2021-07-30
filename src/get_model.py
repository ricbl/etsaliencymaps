# ## Import the model
from .sononet_baseline import Sononet
from .ag_sononet import AG_Sononet
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_model(model_name, load_checkpoint_d):
    if(model_name=='sononet'):
        model = Sononet().to(device)
    elif(model_name=='ag_sononet'):
        model = AG_Sononet().to(device)
    else:
        print('Select models from the following:\n 1) sononet\n 2) ag_sononet')
        assert(False)
    if load_checkpoint_d is not None:
        model.load_state_dict(torch.load(load_checkpoint_d)['state_dict'])
    return model