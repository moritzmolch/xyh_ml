import lightning
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from xyh_ml.config.network import TrainingConfig
from xyh_ml.models.mlp import MultiLayerPerceptron


def import_object(object_path: str):
    module_path, class_name = object_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class PNNClassifier(lightning.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        loss_kwargs,
        optimizer_cls,
        optimizer_kwargs,
        parameter_values,
        parameter_sampling_seed,
    ):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.parameter_values = parameter_values
        self.rng = torch.Generator()
        self.rng.manual_seed(parameter_sampling_seed)

        # Metrics
        self.val_metrics = []

    def forward(self, x):
        return self.model(x)

    def _resample_parameters(self, parameters, resample_parameters):
        # Move the parameter values tensor to the same device as the
        # input parameters tensor
        parameter_values = self.parameter_values.to(parameters.device)

        # Sample random entries from parameter_values
        values_index = torch.randint(
            0,
            parameter_values.shape[0],
            (torch.sum(resample_parameters).int().item(),),
            generator=self.rng,
        ).int()

        # Insert sampled parameter values into parameters tensor, where
        # resample_parameters is True
        parameters[resample_parameters] = parameter_values[values_index]

        return parameters

    def training_step(self, batch, batch_idx):
        # Get batch data and concatenate input and target tensors
        input, parameters, target, weight, resample_parameters = batch
        parameters = self._resample_parameters(parameters, resample_parameters)
        input = torch.cat((input, parameters), dim=-1)
        output = self.model(input)
        loss = self.loss_fn(output, target, weight, **self.loss_kwargs)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, logger=False
        )
        self.logger.log_metrics(
            {"train_loss": loss},
            step=self.global_step,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Get batch data and concatenate input and target tensors
        input, parameters, target, weight, resample_parameters = batch
        parameters = self._resample_parameters(parameters, resample_parameters)
        input = torch.cat((input, parameters), dim=-1)
        output = self.model(input)
        loss = self.loss_fn(output, target, weight, **self.loss_kwargs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False)
        self.val_metrics.append({"val_loss": loss})
        return loss

    def on_validation_epoch_end(self):
        # Log the average validation loss for the epoch
        avg_metrics = {
            k: torch.stack(
                [x[k] for x in self.val_metrics for k in x.keys() if k == k]
            ).mean()
            for k in self.val_metrics[0].keys()
        }
        self.logger.log_metrics(
            avg_metrics,
            step=self.global_step,
        )
        self.val_metrics.clear()

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(), **self.optimizer_kwargs
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


def create_pnn_model(
    input_size: int,
    parameter_size: int,
    output_size: int,
    training_config: TrainingConfig,
    parameter_sampling_values: torch.Tensor,
    parameter_sampling_seed: int | None = None,
):
    # Create the bare pytorch model
    model = MultiLayerPerceptron(
        input_size=input_size + parameter_size,
        output_size=output_size,
        hidden_layers=training_config.hidden_layers,
        output_layer=training_config.output_layer,
    )

    # Import the optimizer class and the loss function
    optimizer_cls = import_object(training_config.optimizer.class_name)
    loss_fn = import_object(training_config.loss.fn_name)

    # Create the lightning wrapper
    lightning = PNNClassifier(
        model=model,
        loss_fn=loss_fn,
        loss_kwargs=training_config.loss.kwargs,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=training_config.optimizer.kwargs,
        parameter_values=parameter_sampling_values,
        parameter_sampling_seed=parameter_sampling_seed,
    )

    return lightning
