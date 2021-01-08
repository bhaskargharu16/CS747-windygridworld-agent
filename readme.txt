Instructions to run the code -
    1. To generate the data for the assignment use the command line option --generatedata
        use command -  'python mainscript.py --generatedata y'
        a. this will generate 4 plots  {1.png,2.png,3.png,4.png}
        b. the first 3 plots are for the sarsa(0) agent in non stochastic no king moves environment, 
        non stochastic king moves environment and stochastic king moves environment respectively 
        c. and the 4th plot is for comparison of sarsa, Qlearning, expected sarsa in non stochastic
            no king moves environment
        Note - all 4 plots are averaged over 50 seeds
        
    2. other command line options are included to run desired algorithm in environments as follows
        a.  --stochastic  
            1. use argument y or n to activate or deactivate respectively
        b. --algorithm 
            1. use one of the following arguments sarsa, qlearning , expectedsarsa
        c. --kings
            1. use argument y or n to choose between 8 or 4 moves respectively for agent
        
        Commands -:
            1. python mainscript.py --stochastic n --algorithm sarsa --kings n
            2. python mainscript.py --stochastic n --algorithm qlearning --kings n
            3. python mainscript.py --stochastic n --algorithm expectedsarsa --kings n
            4. python mainscript.py --stochastic n --algorithm sarsa --kings y
            5. python mainscript.py --stochastic n --algorithm qlearning --kings y 
            6. python mainscript.py --stochastic n --algorithm expectedsarsa --kings y
            7. python mainscript.py --stochastic y --algorithm sarsa --kings y
            8. python mainscript.py --stochastic y --algorithm qlearning  --kings y
            9. python mainscript.py --stochastic y --algorithm expectedsarsa  --kings y
           10. python mainscript.py --stochastic y --algorithm sarsa  --kings n 
           11. python mainscript.py --stochastic y --algorithm qlearning  --kings n
           12. python mainscript.py --stochastic y --algorithm expectedsarsa  --kings n
        
        A plot (timesteps vs episodes) will be saved in 'results.png' which is averaged over 50 seeds
    