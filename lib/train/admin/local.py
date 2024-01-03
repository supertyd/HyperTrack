class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/ubuntu/Downloads/hypertrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.got10k_dir = '/home/ubuntu/Downloads/GOT-10k/train'
        self.whisper_nir_dir = '/home/ubuntu/Downloads/challenge2023/datasets/training/HSI-NIR'
        self.whisper_vis_dir = "/home/ubuntu/Downloads/challenge2023/datasets/training/HSI-VIS"
        self.whisper_rednir_dir = "/home/ubuntu/Downloads/challenge2023/datasets/training/HSI-RedNIR"
