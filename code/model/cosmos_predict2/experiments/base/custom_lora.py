"""
Custom LoRA Fine-Tuning Experiment for V2V
"""
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# We use the pre-trained checkpoint as the base
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Custom dataset
custom_lora_dataset = L(VideoDataset)(
    dataset_dir="/root/autodl-tmp/datasets/custom_lora",
    num_frames=49,            # Generate 49 frames total
    video_size=(352, 640),    # Train at your specific resolution
)

custom_lora_dataloader = L(get_generic_dataloader)(
    dataset=custom_lora_dataset,
    sampler=L(get_sampler)(dataset=custom_lora_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

# Custom experiment
predict2_video2world_training_2b_custom_lora = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="custom_lora",
    ),
    dataloader_train=custom_lora_dataloader,
    checkpoint=dict(
        save_iter=200,
        load_path=DEFAULT_CHECKPOINT.s3.uri,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=2 ** (-15), # Slightly lower LR for LoRA fine-tuning
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[100],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=10,
        max_iter=1000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=200, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=200, save_s3=False),
            every_n_sample_ema=dict(every_n=200, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            use_lora=True,           # Enable LoRA
            lora_rank=16,
            lora_alpha=16,
            init_lora_weights=True,
            
            # Use 5 latent frames as condition (approx 17 pixel frames out of the 49)
            min_num_conditional_frames=5,
            max_num_conditional_frames=5,
            
            text_encoder_config=dict(
                compute_online=False, # 关闭在线推理，防止 32G 显存 OOM
            ),
        )
    ),
)

cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    predict2_video2world_training_2b_custom_lora,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
