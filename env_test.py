import gym
import gym_navigation

env = gym.make('Navigation10x10-v0')

_ = env.reset
env.render()
done = False
env.reset()

while True:
	action = env.action_space.sample()
	_, reward, done, _ = env.step(action)
	print(reward)

	env.render()

	if done:
		break
