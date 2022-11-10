from pytorch_lightning.utilities.cli import LightningCLI
from lightning_modules.models.gpt_stmt_codegen_model import GptStmtCodeGenModel
from lightning_modules.datasets.nb_stmt_reader import NbCellDataModule

from pytorch_lightning import Trainer

trainer = Trainer(gpus=1)

model = GptStmtCodeGenModel.load_from_checkpoint('amlt/pl-nb-full-smaller-bs/train-full-parameterized_gpu16/ckpts/' + \
                                                    'epoch=0-perplexity=20518934.00-cell_edit_dist=283.53.ckpt',
                                                    transformer_model_name='EleutherAI/gpt-neo-125M',
                                                    max_stmt_len = 100,
                                                    max_stmt_num = 10,
                                                    beam_size = 1,
                                                    max_context_len = 412,
                                                    )
datamodule = NbCellDataModule(batch_size=2, 
                              val_batch_size=8, 
                              train_file_path='data/original_nb_stmt/train_shards/train_shard_0.jsonl', 
                              val_file_path='data/original_nb_stmt/val_shards/val_shard_0.jsonl', 
                              train_max_instances=10, 
                              val_max_instances=100,
                              max_context_tokens = 412)

val_results = trainer.validate(model, datamodule, verbose=True)

print(val_results)



