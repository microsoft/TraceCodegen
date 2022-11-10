# import sys
# sys.path.append('/home/t-ansongni/Code/trace-codegen/lightning_modules/')

import pytorch_lightning as pl

from .models.gpt_stmt_codegen_model import GptStmtCodeGenModel
from .datasets.nb_stmt_reader import NbCellDataModule



if __name__ == "__main__":
    model = GptStmtCodeGenModel(transformer_model_name="EleutherAI/gpt-neo-125M", max_stmt_num=10)

    data_module = NbCellDataModule(batch_size=2, 
                                   train_file_path="/home/t-ansongni/Code/trace-codegen/" \
                                        "data/original_nb_stmt/train_shards/train_shard_0.json",
                                   val_file_path="/home/t-ansongni/Code/trace-codegen/" \
                                        "data/original_nb_stmt/val_shards/val_shard_0.json",)

    trainer = pl.Trainer(gpus=1)

    trainer.fit(model, data_module)
