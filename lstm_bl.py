import pandas as pd
import yfinance as yf

start_date = "2003-10-01"
end_date = "2023-05-01"
sp500 = yf.download(tickers=["^GSPC"], start=start_date, end=end_date, interval="1mo")

bonds = yf.download(tickers=["AGG"], start=start_date, end=end_date, interval="1mo")

bonds.reset_index(inplace=True)
sp500.reset_index(inplace=True)

sp500["ticker"] = "^GSPC"
bonds["ticker"] = "AGG"

data = pd.concat([bonds, sp500], axis=0)

data.sort_values(["Date", "ticker"], inplace=True)

test_data = data[data.Date >= "2022"]
train_data = data[data.Date < "2022"]

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import config_params
from black_litterman_env import BlackLittermanEnv
from ray.air.integrations.wandb import WandbLoggerCallback
from drllibv2 import DRLlibv2
import config_params
# parser = argparse.ArgumentParser(description="If confidence output")
# parser.add_argument(
#     "-if", "--if_confidence", type=bool, help="Whether to output confidence",default=True
# )
# args = parser.parse_args()

# stock_env = BlackLittermanEnv(
#     data_df=data,
# )
# print("Reset")
# print(stock_env.reset())


# state,reward,terminal,_,info = stock_env.step([(0.1,0.2,0.05),(0.3,0.4,0.5)])

from ray.tune.registry import register_env

def run_lstm_bl(if_confidence):
    if if_confidence=="true":
        if_confidence = True
    elif if_confidence=="false":
        if_confidence=False
    env_name = "BlackLitterManEnv-v1"
    register_env(
        env_name,
        lambda config: BlackLittermanEnv(
            data_df=train_data, if_confidence=if_confidence
        ),
    )

    train_env_instance = BlackLittermanEnv(train_data, if_confidence=if_confidence)

    


    def sample_ppo_params():
        return {
            # "entropy_coeff": tune.loguniform(0.00000001, 1e-4),
            # "lr": tune.loguniform(5e-5, 0.0001),
            # "sgd_minibatch_size": tune.choice([32, 64, 128, 256]),
            # "lambda": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
             "entropy_coeff": 0.0000001,
              "lr": 5e-5,
              "sgd_minibatch_size": 64,
              "lambda":0.9,
            "framework": "torch",
            "model": {
                "use_lstm": True,
                # "lstm_cell_size": tune.choice([128, 256, 512])
                'lstm_cell_size':256
            },
            "num_envs_per_worker":config_params.num_envs_per_worker
        }


    model_name = "PPO"
    metric = "episode_reward_mean"
    mode = "max"

    search_alg = OptunaSearch(metric=metric, mode=mode)

    scheduler_ = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=config_params.training_iterations,
        grace_period=config_params.training_iterations//10,
        reduction_factor=2,
    )

    

    wandb_callback = WandbLoggerCallback(
        project=config_params.WANDB_PROJECT,
        api_key=config_params.WANDB_API_KEY,
        upload_checkpoints=True,
        log_config=True,
    )
    drl_agent = DRLlibv2(
        trainable=model_name,
        train_env=train_env_instance,
        train_env_name="StockTrading_train",
        framework="torch",
        num_workers=config_params.num_workers,
        log_level="DEBUG",
        run_name="FINRL_TEST_LSTM",
        storage_path="FINRL_TEST_LSTM",
        params=sample_ppo_params(),
        num_samples=config_params.num_samples,
        num_gpus=config_params.num_gpus,
        training_iterations=config_params.training_iterations,
        checkpoint_freq=config_params.checkpoint_freq,
        # scheduler=scheduler_,
        # search_alg=search_alg,
        callbacks=[wandb_callback],
    )

    lstm_res = drl_agent.train_tune_model()

    results_df, best_result = drl_agent.infer_results()


    # checkpoint = lstm_res.get_best_result().checkpoint
    # testing_agent = Algorithm.from_checkpoint(checkpoint)
    results_df.to_csv(f"LSTM_{if_confidence}.csv")
    ds = []
    for i in test_data.groupby("ticker"):
        i[1].reset_index(drop=True, inplace=True)
        ds.append(i[1])
    test_data = pd.concat(ds, axis=0)
    test_data.sort_values(["Date"], inplace=True)

    test_agent = drl_agent.get_test_agent()
    lstm_cell_size = lstm_res.get_best_result().config["model"]["lstm_cell_size"]
    init_state = state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
    import wandb

    wandb.login()

    run = wandb.init(project="Test Data")
    test_table = wandb.Table(dataframe=test_data)

    # Add the table to an Artifact to increase the row
    # limit to 200000 and make it easier to reuse
    # test_table_artifact = wandb.Artifact("test_data_artifact", type="dataset")
    # test_table_artifact.add(test_table, "test_table")
    # run.log({f"LSTM_Test_data": test_table})
    # run.log_artifact(test_table_artifact)

    for runs in range(5):
        test_env_instance = BlackLittermanEnv(test_data, if_confidence=if_confidence)
        test_env_name = "StockTrading_testenv"
        obs = test_env_instance.reset()
        done = False
        while not done:
            action, state, _ = test_agent.compute_single_action(
                observation=obs, state=state
            )
            obs, reward, done, _, _ = test_env_instance.step(action)

        log_df = pd.DataFrame(index=test_data.Date.unique()[3:])
        log_df["Absolute_ret_1"] = [i[0][0] for i in test_env_instance.actions]
        log_df["Confidence_ret_1"] = [i[1][0] for i in test_env_instance.actions]
        log_df["Absolute_ret_2"] = [i[0][1] for i in test_env_instance.actions]
        log_df["Confidence_ret_2"] = [i[1][1] for i in test_env_instance.actions]
        log_df["Relative_ret"] = [i[0][2] for i in test_env_instance.actions]
        log_df["Relative_confidence"] = [i[1][2] for i in test_env_instance.actions]
        log_df["Weights_1"] = [i[0] for i in test_env_instance.weights_memory]
        log_df["Weights_2"] = [i[1] for i in test_env_instance.weights_memory]
        log_df["Assets"] = test_env_instance.asset_memory
        log_df["Portfolio_return"] = test_env_instance.portfolio_return_memory

        log_table = wandb.Table(dataframe=log_df)

        # Add the table to an Artifact to increase the row
        # limit to 200000 and make it easier to reuse

        test_log_artifact = wandb.Artifact("test_log_artifact", type="dataset")
        test_log_artifact.add(log_table, "log_table")

        # Log the table to visualize with a run...
        run.log({f"LSTM_Log_data_{runs}_{if_confidence}": log_table})

        # and Log as an Artifact to increase the available row limit!
        run.log_artifact(test_log_artifact)
