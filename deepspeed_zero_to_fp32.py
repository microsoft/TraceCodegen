from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# lightning deepspeed has saved a directory instead of a file
model_ckpt_dir = "lightning_logs/version_0/checkpoints/"
model_ckpt_name = "epoch=0-step=0.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)