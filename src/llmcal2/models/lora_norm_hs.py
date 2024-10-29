
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List, Literal, Optional

import torch
import lightning as L
from litgpt.lora import lora_filter
from lightning.pytorch.trainer.states import RunningStage
from .lora import GenerativeLoRA
    

class GenerativeLMLoRANormHS(L.LightningModule):

    def __init__(
        self,
        gpt: GenerativeLoRA,
        optimizer: Literal["adamw", "sgd"] = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.gpt = gpt
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

        self.best_val_loss = float("inf")
        self.last_val_loss = float("inf")
        self.patience_count = 0

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if self._optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        elif self._optimizer_name == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {self._optimizer_name}")
        return optimizer

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.gpt(idx, input_pos, output_last_hidden_state)

    def _shared_train_val_step(self, batch, batch_idx):
        prompt_ids = batch["prompt_ids"]
        answers_ids = batch["answers_ids"]
        prompt_mask = batch["prompt_mask"]
        labels = batch["label"]
        
        loss = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            class_logprobs = []
            for i, ans_ids in enumerate(answers):
                full_input_ids = torch.cat([input_ids, ans_ids.unsqueeze(0)], dim=1)
                logprobs = self(full_input_ids, None, False)["logits"][:,:-1,:].log_softmax(dim=2)
                index = full_input_ids[:,1:].unsqueeze(2)
                gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
                logprob = gather_logprobs.sum()
                class_logprobs.append(logprob) 
            logits = torch.stack(class_logprobs, dim=0)
            loss = loss + torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0), reduction="sum")
        batch_samples = prompt_ids.shape[0]

        return {f"loss": loss / batch_samples, "cum_loss": loss, "batch_samples": batch_samples}
    
    def training_step(self, batch, batch_idx):
        return self._shared_train_val_step(batch, batch_idx)
    
    def on_train_start(self):
        self.last_global_step = self.global_step
        self.cum_loss = 0.
        self.cum_num_samples = 0

    def on_train_batch_start(self, batch, batch_idx):
        if self.patience_count >= 10 and self.current_epoch > 0:
            self.trainer.should_stop = True
            return -1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.last_global_step == self.global_step:
            self.cum_loss += outputs["cum_loss"]
            self.cum_num_samples += outputs["batch_samples"]
            return

        if self.cum_num_samples == 0:
            return
        
        for logger in self.loggers:
            logger.log_metrics(
                {"train/ce_per_sample": self.cum_loss / self.cum_num_samples},
                step=self.last_global_step,
            )
        self.last_global_step = self.global_step
        self.cum_loss = 0.
        self.cum_num_samples = 0
    
    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(
            Path(self.trainer.default_root_dir) / f"last.ckpt"
        )

    def on_validation_epoch_start(self) -> None:
        self.val_cum_loss = 0.
        self.val_cum_num_samples = 0

    def validation_step(self, batch, batch_idx):
        return self._shared_train_val_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.val_cum_loss += outputs["cum_loss"]
        self.val_cum_num_samples += outputs["batch_samples"]
        
    def on_validation_epoch_end(self):
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            ce = self.val_cum_loss / self.val_cum_num_samples
            self.last_val_loss = ce
            for logger in self.loggers:
                logger.log_metrics({
                    "val/ce_per_sample": ce.item(),
                }, step=self.global_step)

            if ce < self.best_val_loss:
                self.patience_count = 0
                self.best_val_loss = ce
                self.trainer.save_checkpoint(
                    Path(self.trainer.default_root_dir) / "best.ckpt"
                )
            else:
                self.patience_count += 1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = OrderedDict([
            (k, v) for k, v in checkpoint["state_dict"].items() if lora_filter(k,v)
        ])
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["last_val_loss"] = self.last_val_loss
        checkpoint["patience_count"] = self.patience_count

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Add all parameters to state_dict
        for k, v in self.state_dict().items():
            if k not in checkpoint["state_dict"]:
                checkpoint["state_dict"][k] = v
        self.best_val_loss = checkpoint["best_val_loss"]
        self.last_val_loss = checkpoint["last_val_loss"]
        self.patience_count = checkpoint["patience_count"]

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)

    def predict_step(self, batch, batch_idx):
        prompt_ids: torch.Tensor = batch["prompt_ids"]
        prompt_mask: torch.Tensor = batch["prompt_mask"]
        answers_ids: List[List[torch.Tensor]] = batch["answers_ids"]

        last_emb, mean_emb, max_emb = [], [], []
        logits = []
        for input_ids, attention_mask, answers in zip(prompt_ids, prompt_mask, answers_ids):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            T = torch.sum(attention_mask)
            input_pos = torch.arange(0, T, device=input_ids.device, dtype=input_ids.dtype)
            output = self(input_ids, input_pos, True)
            answers_logits = []
            for answer in answers:
                answer = answer.unsqueeze(0)
                input_pos = torch.arange(T, answer.shape[1] + T, device=answer.device, dtype=answer.dtype) 
                ans_out = self(answer, input_pos, False)
                logprobs = torch.cat([output["logits"][:,-1:,:], ans_out["logits"][:,:-1,:]], dim=1).log_softmax(dim=2)
                index = answer.unsqueeze(2)
                gather_probs = torch.gather(logprobs, -1, index).squeeze(2)
                ans_logit = gather_probs.sum()
                answers_logits.append(ans_logit)
            logits.append(torch.stack(answers_logits, dim=0))
            last_emb.append(output["last_embeddings"][0])
            mean_emb.append(output["mean_embeddings"][0])
            max_emb.append(output["max_embeddings"][0])
        logits = torch.stack(logits, dim=0)
        last_emb = torch.stack(last_emb, dim=0)
        mean_emb = torch.stack(mean_emb, dim=0)
        max_emb = torch.stack(max_emb, dim=0)

        return {
            "idx": batch["idx"], 
            "logits": logits, 
            "prompt_last_embeddings": last_emb, 
            "prompt_mean_embeddings": mean_emb,
            "prompt_max_embeddings": max_emb,
            "label": batch["label"],
        }

    def on_predict_batch_end(self, outputs, batch, batch_idx) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu().float())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = predict_outputs
        
        
        
