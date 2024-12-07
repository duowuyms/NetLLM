class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def add(self, state, action, next_state, reward, done, info):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def __len__(self):
        return len(self.states)


