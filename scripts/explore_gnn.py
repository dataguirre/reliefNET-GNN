import argparse

import joblib
import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from vulnerability_networks.config import PATH_MODELS, PATH_PROCESSED_DATA
from vulnerability_networks.modeling.train import LightningRankEdgeNet, NetworkDataModule


def template():
    """DONT EDIT THIS, JUST COPY TO OBJECTIVE AND SET AS YOU WANT"""
    num_x_features = 3
    embedding_size = trial.suggest_categorical("embedding_size", [32, 64, 128, 256])
    num_layers_msg = trial.suggest_int("num_layers_msg", 1, 5)
    num_layers_mlp = trial.suggest_int("num_layers_mlp", 1, 5)
    edge_embedding_operator = trial.suggest_categorical("embedding_operator", ["concat", "hadamard", "mean"])
    dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.02)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    msg_passing = trial.suggest_categorical("msg_passing", ["GCN", "GAT"])
    if msg_passing == "GCN":
        gat_v2 = None
        gat_heads = None
    elif msg_passing == "GAT":
        gat_v2 = trial.suggest_categorical("gat_v2", [True, False])
        gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4, 8])

    loss_fn = trial.suggest_categorical("loss_fn", ["MSE", "RANK"])
    loss_phi_fn = trial.suggest_categorical("loss_phi_fn", ["sqrt_exp", "exponential"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)


def objective(trial: optuna.trial.Trial, max_epochs, index_accessibility, path_data, path_logging) -> float:
    # modify from your template
    num_x_features = 3
    embedding_size = trial.suggest_categorical("embedding_size", [64])
    num_layers_msg = trial.suggest_int("num_layers_msg", 2, 3)
    num_layers_mlp = trial.suggest_int("num_layers_mlp", 1, 2)
    edge_embedding_operator = "concat" #trial.suggest_categorical("embedding_operator", ["concat", "hadamard", "mean"])
    dropout = trial.suggest_float("dropout", 0.22, 0.3, step=0.02)
    batch_size = trial.suggest_categorical("batch_size", [32])
    msg_passing = trial.suggest_categorical("msg_passing", ["GAT"])
    if msg_passing == "GCN":
        gat_v2 = None
        gat_heads = None
    elif msg_passing == "GAT":
        gat_v2 = trial.suggest_categorical("gat_v2", [True])
        gat_heads = trial.suggest_categorical("gat_heads", [8])

    loss_fn = "RANK" # trial.suggest_categorical("loss_fn", ["MSE", "RANK"])
    loss_phi_fn = "sqrt_exp" # trial.suggest_categorical("loss_phi_fn", ["sqrt_exp", "exponential"])
    lr = trial.suggest_float("lr", 7e-3, 9e-3, log=True)

    L.seed_everything(42, workers=True)

    lit_model = LightningRankEdgeNet(
        num_x_features=num_x_features,
        embedding_size=embedding_size,
        num_layers_msg=num_layers_msg,
        num_layers_mlp=num_layers_mlp,
        msg_passing=msg_passing,
        loss_fn=loss_fn,
        dropout=dropout,
        edge_embedding_operator=edge_embedding_operator,
        loss_phi_fn=loss_phi_fn,
        lr=lr,
        v2=gat_v2,
        heads=gat_heads,
    )
    # train_loader, val_loader, test_loader = get_network_loaders(path_data, batch_size)
    datamodule = NetworkDataModule(data_dir=path_data, batch_size=batch_size)
    tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=path_logging, name=index_accessibility)
    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=path_logging, name=index_accessibility, version=tb_logger.version)
    objective_metric = "val_map_at_k_metric"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        log_every_n_steps=10,
        logger=[tb_logger, csv_logger],
        # enable_checkpointing=False,
        # callbacks=[L.pytorch.callbacks.EarlyStopping(monitor=f"val_loss_{loss_fn}", mode="min", patience=10)],
        deterministic=True,
    )
    trainer.logger.log_hyperparams(lit_model.hparams)
    trainer.fit(lit_model, datamodule)

    return trainer.callback_metrics[objective_metric].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning Optuna Exploration.")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument(
        "--accessibility-index",
        type=str,
        choices=["global_efficiency", "independent_path"],
        default="global_efficiency",
        help="Index accessibility type",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials to run")
    parser.add_argument(
        "--stage", type=str, default="explore", choices=["explore", "refine", "validate"], help="Stage of experimentation"
    )

    # parameters
    args = parser.parse_args()
    max_epochs = args.max_epochs
    index_acc = args.accessibility_index
    n_trials = args.n_trials
    stage = args.stage
    if stage == "explore":
        assert n_trials > 100
        assert max_epochs < 15
    elif stage == "refine":
        assert max_epochs > 25

    path_data = PATH_PROCESSED_DATA / f"{index_acc}_old"

    pruner = optuna.pruners.MedianPruner()
    path_optuna_study = PATH_MODELS / f"{stage}/{index_acc}/optuna_study.pkl"
    path_logging = PATH_MODELS / stage

    sampler = optuna.samplers.TPESampler(seed=42)
    study = (
        joblib.load(path_optuna_study)
        if (path_optuna_study).exists()
        else optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    )
    # objective = create_objective(max_epochs, index_acc, batch_size, path_data)
    study.optimize(lambda trial: objective(trial, max_epochs, index_acc, path_data, path_logging), n_trials=n_trials)
    joblib.dump(study, path_optuna_study)
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
