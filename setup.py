from setuptools import setup, find_packages

setup(name='openai-envs-taxi',
      version='2.0.5',
      description='Custom environments for OpenAI Gym',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/e-dzia/openai-envs',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'gym>=0.10',
          'networkx==2.0',
          'bitstring==3.1.5'
      ],
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
