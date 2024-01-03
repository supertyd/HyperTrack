from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = '/home/ubuntu/Downloads/hypertrack/test/networks'    # Where tracking networks are stored.

    settings.prj_dir = '/home/ubuntu/Downloads/hypertrack'
    settings.result_plot_path = '/home/ubuntu/Downloads/hypertrack/test/result_plots'
    settings.results_path = '/home/ubuntu/Downloads/hypertrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ubuntu/Downloads/hypertrack'

    settings.whisper_vis = '/home/ubuntu/Downloads/challenge2023/datasets/validation/HSI-VIS'
    settings.whisper_nir = '/home/ubuntu/Downloads/challenge2023/datasets/validation/HSI-NIR'
    settings.whisper_rednir = "/home/ubuntu/Downloads/challenge2023/datasets/validation/HSI-RedNIR"
    #settings.whisper_path = "/home/ubuntu/Downloads/challenge2023/datasets/validation/HSI-NIR"
    settings.imec25 = "/media/ubuntu/b47114be-f454-4377-bf84-0e81da2a42bc/IMEC25Dataset"


    return settings
