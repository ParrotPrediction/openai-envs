# LCS environment
The repository contains environments used in LCS literature that are compliant with OpenAI Gym interface.

[![Build Status](https://travis-ci.org/ParrotPrediction/openai-envs.svg?branch=master)](https://travis-ci.org/ParrotPrediction/openai-envs)

For usage examples look at [examples/](examples) directory.

## Development

    conda create --name openai-envs python>=3.7
    conda activate openai-envs

    pip install -e ".[testing]"
    
    python setup.py test
