# Parrot Prediction OpenAI environments

    conda create --name openai-envs python=3.7
    source activate openai-envs

    pip install -r requirements.txt

## Maze

Initializing

    maze = gym.make('MazeF1-v0')

Getting all possible transitions

    transitions = maze.env.get_all_possible_transitions()

## Boolean Multiplexer
Read blog [post](https://medium.com/parrot-prediction/boolean-multiplexer-in-practice-94e3236821b5) describing the usage.