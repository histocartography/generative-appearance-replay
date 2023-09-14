class Config:
    """Class for train/test configuration."""

    def __init__(self, config_dict):
        """Initialize configuration.

        Args:
            config_dict: dictionary of configuration parameters
        """
        # paths
        self.save_path: str = config_dict.get("save_path", "./results")
        self.seg_model_path: str = config_dict.get("seg_model_path", None)
        self.gen_model_path: str = config_dict.get("gen_model_path", None)
        self.ckpt: str = config_dict.get("ckpt", None)

        # task
        self.task: str = config_dict.get("task", None)
        self.n_domains: int = config_dict.get("n_domains", 3)
        self.domain_id: int = config_dict.get("domain_id", 0)

        # data
        self.curr_dataset: str = config_dict.get("curr_dataset", None)
        self.prev_dataset: str = config_dict.get("prev_dataset", None)
        self.src_dataset: str = config_dict.get("src_dataset", None)
        self.curr_split: str = config_dict.get("curr_split", None)
        self.prev_split: str = config_dict.get("prev_split", None)
        self.src_split: str = config_dict.get("src_split", None)
        self.num_workers: int = config_dict.get("num_workers", 8)
        self.batch_size: int = config_dict.get("batch_size", 16)
        self.gan_batch_size: int = config_dict.get("gan_batch_size", 16)
        self.augmentation: bool = config_dict.get("augmentation", False)

        # early stopping
        self.early_stopping: bool = config_dict.get("early_stopping", False)
        self.patience: int = config_dict.get("patience", 10)
        self.stop_epoch: int = config_dict.get("stop_epoch", 50)
        self.stop_metric: str = config_dict.get("stop_metric", "loss")

        # segmentation
        self.n_seg_classes: int = config_dict.get("n_seg_classes", 2)
        self.background_label: int = config_dict.get("background_label", 4)
        self.n_seg_steps: int = config_dict.get("n_seg_steps", 1000)
        self.seg_lr: float = config_dict.get("seg_lr", 0.0005)
        self.kd_temp: float = config_dict.get("kd_temp", 2.0)
        self.optimizer: str = config_dict.get("optimizer", "SGD")

        # GAN
        self.latent_dim: int = config_dict.get("latent_dim", 512)
        self.n_gen_steps: int = config_dict.get("n_gen_steps", 300000)
        self.disc_lr: float = config_dict.get("disc_lr", 0.0002)
        self.gen_lr: float = config_dict.get("gen_lr", 0.0002)
        self.feat_matching_layer: str = config_dict.get("feat_matching_layer", "stage0")
        self.image_size: int = config_dict.get("image_size", 256)
        self.ema_nk_img: int = config_dict.get("ema_hl", 500)
        self.r1_gamma: float = config_dict.get("r1_gamma", None)
        self.r1_interval: int = config_dict.get("r1_interval", 16)
        self.cont_coefficient: float = config_dict.get("cont_coefficient", 1.0)
        self.img_coefficient: float = config_dict.get("img_coefficient", 0.0)
        self.img_dist_loss_type: str = config_dict.get("img_dist_loss_type", "L1")

        # resume training
        self.resume_training: bool = config_dict.get("resume_training", False)

        # seed
        self.seed: int = config_dict.get("seed", 0)
