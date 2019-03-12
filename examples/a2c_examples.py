import gym
import os
import sys
import pickle
import time
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_critic import Value
from core.agent import Agent
from core.common import estimate_advantages
from core.a2c import  a2c_step

parser = argparse.ArgumentParser(description="Synchronous Actor Critic")
parser.add_argument("--config-file", type=str)
args = parser.parse_args()
exp_args = ReadExpConfig(args.config_file)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)

""" environment """
env = gym.make(exp_args["config"]["env-name"])
state_dim = env.observation_space.shape[0]
is_discrete_action_space = len(env.action_space.shape) == 0 # shape is empty () for discrete environments
running_state = ZFilter((state_dim, ), clip=5)

""" Seeding """
np.random.seed(exp_args["config"]["seed"])
torch.manual_seed(exp_args["config"]["seed"])
env.seed(exp_args["config"]["seed"])


""" define policy(actor) and critic(value function predictor) """

if is_discrete_action_space:
	policy_net = DiscretePolicy(state_dim, env.action_space.n, exp_args["model"]["hidden"], exp_args["model"]["activation"])
else:
	raise ValueError("Policy for Continous Action Space is not implemented yet")

value_net = Value(state_dim, exp_args["model"]["hidden"], exp_args["model"]["activation"])

policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=exp_args["config"]["lr"])
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=exp_args["config"]["lr"])

""" Create Agent """

agent = Agent(env, policy_net, device, running_state=running_state, render=exp_args["config"]["render"], num_threads=exp_args["config"]["num-threads"], horizon=exp_args["config"]["horizon"])

agent.collect_samples(2048)

def update_params(batch):

	states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
	actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
	rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
	masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
	
	with torch.no_grad():
		values = value_net(states)

	advantages, returns = estimate_advantages(rewards, masks, values, exp_args["config"]["gamma"], exp_args["config"]["tau"], device)

	a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, exp_args["config"]["l2-reg"])


def main_loop():

	for i_iter in range(exp_args["config"]["max-iter-num"]):

		batch, log = agent.collect_samples(exp_args["config"]["min-batch-size"])
		t0 = time.time()
		update_params(batch)
		t1 = time.time()

		if i_iter % exp_args["config"]["log-interval"] == 0:
			print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

		if exp_args["config"]["save-model-interval"] > 0 and (i_iter+1) % exp_args["config"]["save-model-interval"] == 0:
			to_device(torch.device('cpu'), policy_net, value_net)
			pickle.dump((policy_net, value_net, running_state),open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(exp_args["config"]["env-name"])), 'wb'))
			to_device(device, policy_net, value_net)


		torch.cuda.empty_cache()


main_loop()