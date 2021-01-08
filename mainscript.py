from non_stochastic_4moves import *
from non_stochastic_8moves import *
from stochastic_8moves import *
from stochastic_4moves import *
import numpy as np
import argparse

height = 7
width = 10
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
start_state = (3,0)
goal_state = (3,7)
reward = -1

def experiment1():
    """
    Sarsa(0) agent in Non stochastic no king moves
    """
    print("Running : Sarsa(0) agent in Non stochastic no king moves")
    env1 = WindyGridWorld(width,height,wind,start_state,goal_state,reward)
    env1.populate_destinations()
    final_data1 = np.zeros(8000)
    for i in range(50):
        np.random.seed(i)
        solver = Solver(env1,0.1,0.5)
        solver.sarsa()
        final_data1 += np.array(solver.data)
    final_data1 = final_data1/50

    """
    Sarsa(0) agent in Non stochastic, king moves
    """
    print("Running : Sarsa(0) agent in Non stochastic, king moves")
    env2 = KingsWindyGridWorld(width,height,wind,start_state,goal_state,reward)
    env2.populate_destinations()

    final_data2 = np.zeros(8000)

    for i in range(50):
        np.random.seed(i)
        solver = KingsSolver(env2,0.1,0.5)
        solver.sarsa()
        final_data2 += np.array(solver.data)

    final_data2 = final_data2/50

    """
    Sarsa(0) agent in stochastic king moves
    """
    print("Running : Sarsa(0) agent in stochastic, king moves")
    env3 = StochasticKingsWindyWorld(width,height,wind,start_state,goal_state,reward)

    final_data3 = np.zeros(8000)

    for i in range(50):
        np.random.seed(i)
        solver = StochasticKingsSolver(env3,0.1,0.5)
        solver.sarsa()
        final_data3 += np.array(solver.data)

    final_data3 = final_data3/50

    plt.figure()
    plt.plot(np.arange(8000),final_data1)
    plt.xlabel('timestep')
    plt.ylabel('episodes')
    plt.legend(['non stochastic'])
    plt.title("Sarsa(0) agent")
    plt.savefig("1.png")

    plt.figure()
    plt.plot(np.arange(8000),final_data2)
    plt.xlabel('timestep')
    plt.ylabel('episodes')
    plt.legend(['non stochastic, king moves'])
    plt.title("Sarsa(0) agent")
    plt.savefig("2.png")

    plt.figure()
    plt.plot(np.arange(8000),final_data3)
    plt.xlabel('timestep')
    plt.ylabel('episodes')
    plt.legend(['stochastic, king moves'])
    plt.title("Sarsa(0) agent")
    plt.savefig("3.png")

def experiment2():
    """
    Algorithm comparison for non stochastic and non king moves
    """
    print("Algorithm comparison for non stochastic and non king moves : -")

    env1 = WindyGridWorld(width,height,wind,start_state,goal_state,reward)
    env1.populate_destinations()

    final_data1 = np.zeros(8000)
    final_data2 = np.zeros(8000)
    final_data3 = np.zeros(8000)
    print("Running : sarsa")
    for i in range(50):
        np.random.seed(i)
        solver = Solver(env1,0.1,0.5)
        solver.sarsa()
        final_data1 += np.array(solver.data)
    print("Running : qlearning ")
    for i in range(50):
        np.random.seed(i)
        solver = Solver(env1,0.1,0.5)
        solver.q_learning()
        final_data2 += np.array(solver.data)
    print("Running : expected sarsa")
    for i in range(50):
        np.random.seed(i)
        solver = Solver(env1,0.1,0.5)
        solver.expected_sarsa()
        final_data3 += np.array(solver.data)

    final_data1 = final_data1/50
    final_data2 = final_data2/50
    final_data3 = final_data3/50

    plt.figure()
    plt.plot(np.arange(8000),final_data1)
    plt.plot(np.arange(8000),final_data2)
    plt.plot(np.arange(8000),final_data3)
    plt.xlabel('timestep')
    plt.ylabel('episodes')
    plt.legend(['sarsa','Q learning','expected sarsa'])
    plt.title("Non stochastic , non king moves")
    plt.savefig("4.png")

def run_algortihm(stochasticity,algorithm,kings_moves):
    env = None
    solver = None
    final_data = np.zeros(8000)
    if stochasticity == 'y' and kings_moves == 'y':
        env = StochasticKingsWindyWorld(width,height,wind,start_state,goal_state,reward)
        for i in range(50):
            np.random.seed(i)
            solver = StochasticKingsSolver(env,0.1,0.5)
            if algorithm == 'sarsa':
                solver.sarsa()
            elif algorithm == 'qlearning':
                solver.q_learning()
            else:
                solver.expected_sarsa()
            final_data += np.array(solver.data)
    elif stochasticity == 'y' and kings_moves != 'y':
        env = StochasticWindyGridWorld(width,height,wind,start_state,goal_state,reward)
        for i in range(50):
            np.random.seed(i)
            solver = StochasticSolver(env,0.1,0.5)
            if algorithm == 'sarsa':
                solver.sarsa()
            elif algorithm == 'qlearning':
                solver.q_learning()
            else:
                solver.expected_sarsa()
            final_data += np.array(solver.data)
    elif stochasticity != 'y' and kings_moves == 'y':
        env = KingsWindyGridWorld(width,height,wind,start_state,goal_state,reward)
        env.populate_destinations()
        for i in range(50):
            np.random.seed(i)
            solver = KingsSolver(env,0.1,0.5)
            if algorithm == 'sarsa':
                solver.sarsa()
            elif algorithm == 'qlearning':
                solver.q_learning()
            else:
                solver.expected_sarsa()
            final_data += np.array(solver.data)
    else:
        env = WindyGridWorld(width,height,wind,start_state,goal_state,reward)
        env.populate_destinations()
        for i in range(50):
            np.random.seed(i)
            solver = Solver(env,0.1,0.5)
            if algorithm == 'sarsa':
                solver.sarsa()
            elif algorithm == 'qlearning':
                solver.q_learning()
            else:
                solver.expected_sarsa()
            final_data += np.array(solver.data)
    final_data = final_data/50
       
    plt.figure()
    plt.plot(np.arange(8000),final_data)
    plt.xlabel('timestep')
    plt.ylabel('episodes')
    plt.title("{} agent".format(algorithm))
    plt.savefig("result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--stochastic',type=str,default='n')
    parser.add_argument('--algorithm',type=str,default='sarsa')
    parser.add_argument('--kings',type=str,default='n')
    parser.add_argument('--generatedata',type=str,default='n')

    args = parser.parse_args()
    stochasticity = args.stochastic
    algorithm = args.algorithm
    kings_moves = args.kings
    generate_data = args.generatedata

    if generate_data == 'y':
        experiment1()
        experiment2()
    else:
        run_algortihm(stochasticity,algorithm,kings_moves)
