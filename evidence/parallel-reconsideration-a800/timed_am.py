import json
import os
import time
import torch
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.speculative_decoding.train_eagle3 import TrainEagle3Recipe
class TimedRecipe(TrainEagle3Recipe):
    def _maybe_save_step_checkpoint(self, epoch):
        if time.monotonic() - self.pilot_start >= self.pilot_seconds:
            self.total_optim_steps = self.runtime.global_step
            self.pilot_stop_epoch = epoch
        return super()._maybe_save_step_checkpoint(epoch)

    def _maybe_save_final_checkpoint(self, completed_epochs):
        return super()._maybe_save_final_checkpoint(getattr(self, "pilot_stop_epoch", completed_epochs))

    def _wandb_log(self, data, *, step):
        print("PRC_METRIC " + json.dumps(dict(
            data, step=step, elapsed=time.monotonic()-self.pilot_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )), flush=True)

recipe = TimedRecipe(parse_args_and_load_config())
recipe.setup()
torch.save({name: value.detach().cpu() for name, value in recipe.draft_model.state_dict().items()
            if name.startswith(("self_query.", "halt_gate."))}, "/root/prc-pilot/artifacts/am-initial-probes.pt")
torch.cuda.reset_peak_memory_stats()
recipe.pilot_seconds = max(1, int(os.environ.get("PRC_DEADLINE_EPOCH", "0")) - int(time.time()))
recipe.pilot_start = time.monotonic()
print("PRC_AM_TRAIN_START seconds=" + str(recipe.pilot_seconds), flush=True)
recipe.run_train_validation_loop()
print("PRC_AM_DONE " + json.dumps(dict(
    step=recipe.runtime.global_step, elapsed=time.monotonic()-recipe.pilot_start,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
)), flush=True)
