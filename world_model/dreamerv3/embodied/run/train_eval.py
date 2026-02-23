import re

import embodied
import numpy as np


def train_eval(agent, train_env, eval_env, train_replay, eval_replay, logger, args):
    """
    Train an agent using the given environments and replay buffers, and evaluate it on the given evaluation environment.
    Args:
        agent: The agent to train and evaluate.
        train_env: The environment to train the agent on.
        eval_env: The environment to evaluate the agent on.
        train_replay: The replay buffer to store training data in.
        eval_replay: The replay buffer to store evaluation data in.
        logger: The logger to write logs to.
        args: The command-line arguments.
    """
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
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every) # TODO: I feel like this should be Every, not Clock, but danijar seems to only use Clock in his newest update to the repo for some reason, so I will keep it
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
    should_sync = embodied.when.Every(args.sync_every)

    # Initialize the logger, step counter, updates counter, and metrics accumulator
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(train_env.obs_space), sep="\n")
    print("Action space:", embodied.format(train_env.act_space), sep="\n")

    # Initializes a timer to measure execution time for different operations.
    # A list of methods (like "policy", "train") whose execution times will be tracked.
    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", train_env, ["step"])
    if hasattr(train_replay, "_sample"):
        timer.wrap("replay", train_replay, ["_sample"])

    # Define the keys to log for the video
    nonzeros = set()

    def per_episode(ep, mode):

        if mode == "eval":
            print("Evaluating episode")

        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "reward_rate": (ep["reward"] - ep["reward"].min() >= 0.1).mean(),
            },
            prefix=("episode" if mode == "train" else f"{mode}_episode"),
        )

        if mode == "train":
            logger.add(
                {"intrinsic_return": float(ep["intrinsic_reward"].astype(np.float64).sum()),
                },
                prefix="episode",
            )

        # Force the logger to write the episode data
        logger.write()

        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix=f"{mode}_stats")

    # ----- Define the training loop -----
    # Define the driver for training and evaluation
    # The driver is responsible for running the agent in the environment and storing the data in the replay buffer
    # - managing interactions with the training environment.
    driver_train = embodied.Driver(
        train_env,
        obs_intrinsic_reward=args.obs_intrinsic,
        hash_bits=args.hash_bits,
        intr_reward_coeff=args.intr_reward_coeff,
        latents_intrinsic_reward=args.latents_intrinsic,
        ignore_extr_reward=args.ignore_extr_reward,
    )
    # Register: Synchronize environment interaction with logging and replay buffer updates.
    # Registers a callback to execute after every episode
    driver_train.on_episode(lambda ep, worker: per_episode(ep, mode="train"))
    # Registers two callbacks to execute after every environment step
    driver_train.on_step(lambda tran, _: step.increment())
    driver_train.on_step(train_replay.add)

    # The same for the evaluation environment
    driver_eval = embodied.Driver(eval_env)
    driver_eval.on_step(eval_replay.add)
    driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode="eval"))

    # ----- Prefill the replay buffer -----
    # TODO: is this using a random agent to fill the replay buffer?
    # Random agent does not have world model?
    random_agent = embodied.RandomAgent(train_env.act_space)
    print("Prefill train dataset.")
    while len(train_replay) < max(args.batch_steps, args.train_fill):
        driver_train(random_agent.policy, steps=100) # call __call__ method (which is _step method)
    print("Prefill eval dataset.")
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset_train = agent.dataset(train_replay.dataset)
    dataset_eval = agent.dataset(eval_replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset_train"):
                batch[0] = next(dataset_train)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")
            if "priority" in outs:
                train_replay.prioritize(outs["key"], outs["priority"])
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            logger.add(metrics.result())
            logger.add(agent.report(batch[0]), prefix="report")
            with timer.scope("dataset_eval"):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch), prefix="eval")
            logger.add(train_replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="eval_replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver_train.on_step(train_step)

    # A checkpoint object that saves the agent's state and the replay buffers to disk.
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent

    # Load the checkpoint if it exists
    if not args.transfer:
        checkpoint.train_replay = train_replay
        checkpoint.eval_replay = eval_replay

    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)

        if args.transfer:
            # Reset the step counter to zero if transfer 
            # (otherwise it will continue from the step where the checkpoint was saved, 
            # and stop training after args.steps - step.value steps, rather than args.steps)
            step.value = 0

            # Set agent to transfer mode, nesting is necessary for some reason
            agent.agent.set_transfer(True)
            print("Agent is in transfer mode")

    if args.finetune:
        # Reset the step counter to zero if fine-tuning
        # (otherwise it will continue from the step where the checkpoint was saved, 
        # and stop training after args.steps - step.value steps, rather than args.steps)
        step.value = 0

    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    # ----- Training loop -----
    print("Start training loop.")
    # A function that determines the agent's policy during training. It chooses between exploration and training modes.
    policy_train = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    policy_eval = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        # Checks if evaluation is due
        if should_eval(step):
            print("Starting evaluation at step", int(step))
            driver_eval.reset()
            driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
        # Runs the training driver (driver_train) using the training policy (policy_train) for 100 steps at a time.
        driver_train(policy_train, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
    logger.write()
