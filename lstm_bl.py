import yfinance as yf
import pandas as pd

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

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import yfinance as yf

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions
import prettytable
from collections import OrderedDict
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


class BlackLittermanEnv(gym.Env):
    def __init__(
        self,
        data_df: pd.DataFrame,
        information_cols: list = ["Open", "High", "Low", "Adj Close", "Volume"],
        if_confidence: bool = True,
        stock1_weight: float = 0.4,
        stock2_weight: float = 0.6,
        initial_amount: float = 1000_000,
        transaction_cost_pct: float = 0.001,
        base_return: float = 0.02,
        base_return_penalty: float = 2000,
    ):

        self.data_df = data_df

        self.stock_dim = 2
        self.state_space_shape = self.stock_dim
        self.if_confidence = if_confidence
        self.information_cols = information_cols
        self.transaction_cost_pct = transaction_cost_pct
        self.base_return = base_return
        self.base_return_penalty = -base_return_penalty
        self.actions = []
        self.return_action_space = spaces.Box(
            low=-1, high=+1, shape=(self.stock_dim + 1,), dtype=float
        )

        if self.if_confidence:
            self.confidence_action_space = spaces.Box(
                low=0, high=+1, shape=(self.stock_dim + 1,), dtype=float
            )
            self.action_space = spaces.Tuple(
                (self.return_action_space, self.confidence_action_space)
            )
        else:
            self.action_space = self.return_action_space

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.information_cols), self.state_space_shape),
        )

        self.terminal = False
        self.initial_amount = initial_amount
        self.portfolio_value = initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[0] * self.stock_dim]
        self.month = 3
        self.start_month = self.month
        self.data = self.data_df.loc[self.month :]
        self.date_memory = [self.data.Date.unique()[0]]
        self.transaction_cost_memory = []

        self.tickers = self.data_df["ticker"].unique().tolist()
        self.stock1_tic = self.tickers[0]
        self.stock2_tic = self.tickers[1]

        self.stock1_wt = stock1_weight
        self.stock2_wt = stock2_weight
        self.market_caps = {
            self.stock1_tic: self.stock1_wt,
            self.stock2_tic: self.stock2_wt,
        }

        self.stock1_data_close = self.data_df.loc[
            self.data_df["ticker"] == self.stock1_tic
        ]["Adj Close"]
        self.stock2_data_close = self.data_df.loc[
            self.data_df["ticker"] == self.stock2_tic
        ]["Adj Close"]

        self.data_close = pd.DataFrame()
        self.data_close[self.stock1_tic] = self.stock1_data_close.values
        self.data_close[self.stock2_tic] = self.stock2_data_close.values
        self.data_close.reset_index(drop=True, inplace=True)
        self.market_prices = (
            self.stock1_wt * self.stock1_data_close
            + self.stock2_wt * self.stock2_data_close
        )
        self.market_prices.reset_index(drop=True, inplace=True)

    def reset(self, seed=None, options=None):
        self.asset_memory = [self.initial_amount]
        self.month = self.start_month
        self.data = self.data_df.loc[self.month, :]
        self.state = [self.data[ic].values.tolist() for ic in self.information_cols]

        self.state = np.array(self.state)
        self.portfolio_value = self.initial_amount
        self.portfolio_return_memory = [0]

        self.terminal = False
        self.weights_memory = [[0] * self.stock_dim]
        self.transaction_cost_memory = []

        return self.state, {}

    # def black_litterman_weights(self)
    def step(self, actions):

        self.terminal = self.month >= len(self.data_df.Date.unique()) - 1
        # print(self.month,len(self.data_df.index)-1)
        actions = np.array(actions)
        self.actions.append(list(actions))
        # print(actions,actions.shape)

        if self.terminal:
            return self.state, self.reward, self.terminal, self.terminal, {}
        else:
            self.stock1_ret = actions[0][0]
            self.stock2_ret = actions[0][1]
            self.relative_ret = actions[0][2]

            if self.if_confidence:
                self.stock1_conf = actions[1][0]
                self.stock2_conf = actions[1][1]
                self.relative_conf = actions[1][2]
                confidences = [self.stock1_conf, self.stock2_conf, self.relative_conf]

            P_mat = np.array([[1, 0], [0, 1], [1, -1]])

            if self.relative_ret < 0:
                P_mat[2][0] = -1
                P_mat[2][1] = 1
            elif self.relative_ret == 0:
                P_mat[2][0] = 0
                P_mat[2][1] = 0

            Q_mat = np.array(
                [self.stock1_ret, self.stock2_ret, abs(self.relative_ret)]
            ).reshape(-1, 1)

            # print(self.data_close[:1])
            # print(self.data_close[:self.month])
            # S = risk_models.CovarianceShrinkage(self.data_close[:self.month]).ledoit_wolf()
            S = risk_models.CovarianceShrinkage(
                self.data_close[: self.month]
            ).shrunk_covariance()
            # S = risk_models.CovarianceShrinkage(self.data_close[:self.month]).oracle_approximating()
            delta = black_litterman.market_implied_risk_aversion(
                self.market_prices[: self.month]
            )

            market_prior = black_litterman.market_implied_prior_returns(
                self.market_caps, delta, S
            )
            # print(market_prior,S,P_mat,Q_mat)

            if self.if_confidence:
                bl = BlackLittermanModel(
                    S,
                    pi=market_prior,
                    P=P_mat,
                    Q=Q_mat,
                    omega="idzorek",
                    view_confidences=confidences,
                )

            else:
                bl = BlackLittermanModel(S, pi=market_prior, P=P_mat, Q=Q_mat)

            ret_bl = bl.bl_returns()

            S_bl = bl.bl_cov()

            try:

                ef = EfficientFrontier(ret_bl, S_bl)
                ef.add_objective(objective_functions.L2_reg)

                ef.max_sharpe(risk_free_rate=self.base_return)
                weights = ef.clean_weights()
            except:
                self.portfolio_value = (
                    self.data["Adj Close"] * self.weights_memory[-1]
                ).values.sum()
                self.asset_memory.append(self.portfolio_value)
                self.weights_memory.append(self.weights_memory[-1])
                self.reward = self.base_return_penalty
                self.portfolio_return_memory.append(
                    -self.base_return_penalty / self.portfolio_value
                )
                self.month += 1

                self.data = self.data_df.loc[self.month, :]
                self.state = np.array(
                    [self.data[ic].values.tolist() for ic in self.information_cols]
                )
                self.state = np.array(self.state)

                return self.state, self.reward, self.terminal, self.terminal, {}

            weights = pd.Series(weights).values
            self.weights_memory.append(weights)

            last_day_memory = self.data
            self.month += 1

            self.data = self.data_df.loc[self.month, :]
            self.state = np.array(
                [self.data[ic].values.tolist() for ic in self.information_cols]
            )
            self.state = np.array(self.state)
            portfolio_return = sum(
                (
                    (
                        self.data["Adj Close"].values
                        / last_day_memory["Adj Close"].values
                    )
                    - 1
                )
                * weights
            )
            # new_weights = self.normalization([weights[0]] + list(
            #     np.array(weights[1:]) * np.array(
            #         (self.data["Adj Close"].values / last_day_memory["Adj Close"].values))))
            # self.weights_memory.append(new_weights)

            weights_old = self.weights_memory[-2]
            weights_new = self.weights_memory[-1]

            diff_weights = np.sum(np.abs(np.array(weights_old) - np.array(weights_new)))
            transcationfee = (
                diff_weights * self.transaction_cost_pct * self.portfolio_value
            )

            # calculate the overall result
            new_portfolio_value = (self.portfolio_value - transcationfee) * (
                1 + portfolio_return
            )
            portfolio_return = (
                new_portfolio_value - self.portfolio_value
            ) / self.portfolio_value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            # self.date_memory.append(self.data.Date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

        return self.state, self.reward, self.terminal, self.terminal, {}

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.data_df.Date.unique()[self.start_month :]

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = date_list

        return df_return

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.sum(actions)
        actions = actions / sum
        return actions

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.data_df.Date.unique()[self.start_month :]

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = date_list

        return df_value


# stock_env = BlackLittermanEnv(
#     data_df=data,
# )
# print("Reset")
# print(stock_env.reset())


# state,reward,terminal,_,info = stock_env.step([(0.1,0.2,0.05),(0.3,0.4,0.5)])

from ray.tune.registry import register_env

env_name = "BlackLitterManEnv-v1"
register_env(env_name, lambda config: BlackLittermanEnv(data_df=train_data))

train_env_instance = BlackLittermanEnv(train_data)

from drllibv2 import DRLlibv2


def sample_ppo_params():
    return {
        "entropy_coeff": tune.loguniform(0.00000001, 1e-4),
        "lr": tune.loguniform(5e-5, 0.0001),
        "sgd_minibatch_size": tune.choice([32, 64, 128, 256]),
        "lambda": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        #  "entropy_coeff": 0.0000001,
        #   "lr": 5e-5,
        #   "sgd_minibatch_size": 64,
        #   "lambda":0.9,
        "framework": "torch",
        "model": {
            "use_lstm": True,
            "lstm_cell_size": tune.choice([128, 256, 512])
            # 'lstm_cell_size':256
        },
    }


model_name = "PPO"
metric = "episode_reward_mean"
mode = "max"

search_alg = OptunaSearch(metric=metric, mode=mode)

scheduler_ = ASHAScheduler(
    metric=metric,
    mode=mode,
    max_t=5,
    grace_period=1,
    reduction_factor=2,
)
import config_params
from ray.air.integrations.wandb import WandbLoggerCallback

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

from ray.rllib.algorithms import Algorithm

# checkpoint = lstm_res.get_best_result().checkpoint
# testing_agent = Algorithm.from_checkpoint(checkpoint)
results_df.to_csv("LSTM.csv")
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
test_table_artifact = wandb.Artifact("test_data_artifact", type="dataset")
test_table_artifact.add(test_table, "test_table")
run.log({f"LSTM_Test_data": test_table})
run.log_artifact(test_table_artifact)

for runs in range(5):
    test_env_instance = BlackLittermanEnv(test_data)
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
    run.log({f"LSTM_Log_data_{runs}": log_table})

    # and Log as an Artifact to increase the available row limit!
    run.log_artifact(test_log_artifact)
