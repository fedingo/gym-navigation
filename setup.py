from setuptools import setup

setup(name='gym_navigation',
      version='0.1',
      url="https://github.com/fedingo/gym-navigation",
      author="Federico Rossetto",
      license="MIT",
      packages=["gym_navigation", "gym_navigation.envs"],
      install_requires=['gym', 'numpy']
)