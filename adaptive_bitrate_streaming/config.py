import os


class Config:
    _base_dir = '' if 'adaptive_bitrate_streaming' in os.getcwd() else 'adaptive_bitrate_streaming/'
    baseline_model_paths = {
        'genet': _base_dir + 'data/all_models/genet/nn_model_ep_9900.ckpt',
        'udr_1': _base_dir + 'data/all_models/udr_1/nn_model_ep_57600.ckpt',
        'udr_2': _base_dir + 'data/all_models/udr_2/nn_model_ep_52400.ckpt',
        'udr_3': _base_dir + 'data/all_models/udr_3/nn_model_ep_58000.ckpt',
        'udr_real': _base_dir + 'data/all_models/udr_real/nn_model_ep_49000.ckpt',
    }
    
    trace_dirs = {
        'fcc-train': _base_dir + 'data/traces/train/fcc-train/',
        'fcc-valid': _base_dir + 'data/traces/valid/fcc-valid/',
        'fcc-test': _base_dir + 'data/traces/test/fcc-test/',
    }

    video_size_dirs = {
        'video1': _base_dir + 'data/videos/video1_sizes/',
        'video2': _base_dir + 'data/videos/video2_sizes/',
    }

    artifacts_dir = _base_dir + 'artifacts/'
    results_dir = artifacts_dir + 'results/'
    exp_pools_dir = artifacts_dir + 'exp_pools/'

    # plm special
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir = _base_dir + ('../../downloaded_plms' if 'adaptive_bitrate_streaming' in _base_dir else '../downloaded_plms')
    plm_ft_dir = _base_dir + 'data/ft_plms'
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama': {
            'base': 4096,
        },
        't5-lm': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
    }
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama': {
            'base': 32,
        },
        't5-lm': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
    }


cfg = Config()
