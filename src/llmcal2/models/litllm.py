import torch
from litgpt import LLM
import lightning as L

class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, trainer_ckpt_path=None, seed=None):
        super().__init__()
 
        self.llm = llm = LLM.load(
            model=checkpoint_dir,
            init="pretrained",
            distribute="auto",
        )
        self.trainer_ckpt_path = trainer_ckpt_path
        self.random_state = seed

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
    def _shared_train_val_step(self, batch, batch_idx):
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        
        loss = 0
        num_tokens = 0
        for input_ids, attention_mask in zip(prompt_ids, prompt_mask):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            T = input_ids.size(1)
            input_pos = torch.arange(0, T, device=input_ids.device, dtype=input_ids.dtype)
            logprobs = self.llm.model(input_ids, input_pos, False)["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.size(1)

        return {f"loss": loss / num_tokens, "cum_loss": loss, "num_tokens": num_tokens}
    
    def training_step(self, batch, batch_idx):
        return self._shared_train_val_step(batch, batch_idx)
    
    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]