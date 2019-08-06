from setuptools import setup, find_packages

setup(name='parrotprediction-openai-envs',
      version='2.1.0',
      description='Custom environments for OpenAI Gym',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/ParrotPrediction/openai-envs',
      author='Parrot Prediction Ltd.',
      author_email='nkozlowski@protonmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'gym==0.11',
          'networkx==2.0',
          'bitstring==3.1.5'
      ],
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
