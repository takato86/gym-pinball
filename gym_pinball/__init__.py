from gym.envs.registration import register

register(
    id='PinBall-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    nondeterministic = True,
)

register(
    id='InfinitePinBall-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    kwargs={'infinite':1},
    nondeterministic = True,
)

register(
    id='PinBall-Box-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    kwargs={'configuration':1},
    nondeterministic = True,
)


register(
    id='PinBall-Empty-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    kwargs={'configuration':0},
    nondeterministic = True,
)


register(
    id='PinBall-Hard-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    kwargs={'configuration':4},
    nondeterministic = True,
)

register(
    id='PinBall-Medium-v0',
    entry_point='gym_pinball.envs:PinBallEnv',
    kwargs={'configuration':3},
    nondeterministic = True,
)
