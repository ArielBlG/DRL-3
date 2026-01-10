import time


from actor_critic import (
    ActorCriticConfig, 
    UnifiedActorCriticAgent
)
if __name__ == "__main__":
    # env_id = "MountainCarContinuous-v0"
    # env_id = "CartPole-v1"
    env_id = "Acrobot-v1"

    cfg = ActorCriticConfig(
        env_id=env_id,
        gamma=0.996,
        total_episodes=400,
        entropy_coef=0.01,
        log_every=10,
        seed=0,
        device="cpu",
        log_to_tensorboard=True,
        grad_clip_norm=None,
        policy_lr=3e-4,
        value_lr=1e-3,
        n_bins=3,
        policy_hidden_sizes=[64,64],
        value_hidden_sizes=[64,64],

    )

    # agent = UnifiedActorCriticAgent(cfg)
    # cfg.env_id = "Acrobot-v1"
    cfg.env_id = "CartPole-v1"
    agent_source = UnifiedActorCriticAgent(cfg)
    agent_source.train()
    agent_source.save_model("acrobot_source.pt")
    agent_source.close()
    try:
        print("Done training source, not training target")
        t0 = time.perf_counter()
        # cfg.env_id = "CartPole-v1"
        cfg.env_id = "MountainCarContinuous-v0"
        cfg.total_episodes = 500
        agent_target = UnifiedActorCriticAgent(cfg)
        agent_target.load_transfer_model("acrobot_source.pt", freeze_hidden=False)
        agent_target.train()
        print(f"took {time.perf_counter() - t0:.6f} seconds")
        print("Eval avg return:", agent_target.evaluate(episodes=20, deterministic=True))
        print(f"(Unified dims) obs_dim_max={agent_target.obs_dim_max}, policy_out_dim={agent_target.policy_out_dim}")
    except Exception as e:
        raise(e)
    finally:
        agent_target.close()