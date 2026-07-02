"""Absolute-skill benchmark: play a trained brain (greedily) against the fixed bots
and report win-rate. Works for either algorithm because both agents have act()."""
import torch
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import EasyBot, MediumBot, HardBot
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent

def _load(weights, algo):
    agent = PPOAgent() if algo == "ppo" else DQNAgent()
    agent.load(weights)
    return agent

def evaluate(weights, role, algo="ppo", bots=(EasyBot, MediumBot, HardBot), n_games=100):
    agent = _load(weights, algo)
    scores = {}
    for bot in bots:
        env = ArenaBrawlEnv(agent_role=role(), opponent_role=role(), opponent_bot=bot)
        wins = 0
        for _ in range(n_games):
            s, _ = env.reset()
            done = False
            while not done:
                s, r, term, trunc, _ = env.step(agent.act(s))   # greedy; DQN or PPO, same call
                done = term or trunc
            if term and env.opponent.hp <= 0:
                wins += 1
        env.close()
        scores[bot.__name__] = wins / n_games
    print(f"{algo.upper()} {role.__name__}:  " + "   ".join(f"{k} {v:.0%}" for k, v in scores.items()))
    return scores

if __name__ == "__main__":
    # example — evaluate a trained PPO Gunner
    evaluate("ai/weights/ppo_gunner_final.pth", Gunner, algo="ppo", n_games=50)
