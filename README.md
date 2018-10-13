# LCS environment
The repository contains environments used in LCS literature that are compliant with OpenAI Gym interface.

## Currently implemented environments

- Maze (different variants)
- Binary Multiplexer
- Real Multiplexer
- Hand Eye

For usage examples look at [examples/](examples) directory.

## Development

    conda create --name openai-envs python=3.7
    source activate openai-envs

    pip install -r requirements.txt
    
    make test