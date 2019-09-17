import gym
import gym_navigation

env = gym.make('Navigation10x10-v0')

_ = env.reset()
env.render()
done = False

while True:
	_, reward, done, _ = env.step(env.action_space.sample())
	print(reward)

	env.render()

	if done:
		break