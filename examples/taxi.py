import gym
import gym_taxi_goal


if __name__ == '__main__':
    # Load desired environment
    environment = gym.make('TaxiGoal-v0')

    situation = environment.reset()
    environment.render()
    done = False

    for i in range(100):
        situation = environment.reset()
        environment.render()
        print("\ns:", list(environment.env.decode(situation)))
        print("g:", list(environment.env.decode(
            environment.env.get_goal_state())))
        # environment.render()
