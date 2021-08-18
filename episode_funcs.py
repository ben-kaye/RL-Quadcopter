from tf_agents.trajectories import trajectory

def collect_episode(environment, policy, replay_buffer, num_episodes=10):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)

        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)
        if traj.is_boundary():
            episode_counter += 1

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)

            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return/num_episodes
    return avg_return.numpy()[0]



