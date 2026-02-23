import re

import embodied
import numpy as np


def train(agent, env, replay, logger, args):
    # ----- Initialization -----
    # Create and configure the logging directory
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)

    # Define schedules and conditions for:
    # Exploration (should_expl).
    # Training (should_train).
    # Logging (should_log).
    # Model saving (should_save).
    should_expl = embodied.when.Until(
        args.expl_until
    )  # TODO: args.expl_until is 0 so it is always exploring?
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)

    # Initialize the logger, step counter, updates counter, and metrics accumulator
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    # Initializes a timer to measure execution time for different operations.
    # A list of methods (like "policy", "train") whose execution times will be tracked.
    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):  # ep is a collection of obs and acts in this episode?
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(
                args.log_keys_sum, key
            ):  # Only true when the key is an empty string?
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    # ----- Define the training loop -----
    # Define the driver for training and evaluation
    # The driver is responsible for running the agent in the environment and storing the data in the replay buffer
    # - managing interactions with the training environment.
    driver = embodied.Driver(env)  # TODO: we should add intrinsic rewards
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space)
    while len(replay) < max(args.batch_steps, args.train_fill):
        # ep keys ['image', 'reward', 'intrinsic_reward', 'is_first', 'is_last', 'is_terminal', 'action', 'reset']
        # TODO:
        # Q: why the time steps are ~641; when it is ~640, it calls the on_episode function -> it has a last state?
        # Q: reward and intrinsic_reward all zero -> reward = extrinsic reward?
        # Q: image is almost always 0 -> is it because until now there is no training involved
        #   -> but it is observation it should be from the environment
        # action is a series of one-hot encoded actions
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    # Prepare the data from the random agent
    dataset = agent.dataset(
        replay.dataset
    )  # JAXAgent.dataset; add replay dataset as a generator
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")
            if "priority" in outs:
                replay.prioritize(outs["key"], outs["priority"])
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    # Load or save the checkpoint
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    # ----- Run the training loop -----
    print("Start training loop.")
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
