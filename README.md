# CoachAI+ Badminton Environment

- This project is part of the Coach AI Badminton Project from the [Advanced Database System Laboratory](https://lab-adsl-website.vercel.app/) at National Yang Ming Chiao Tung University, supervised by Prof. [Wen-Chih Peng](https://sites.google.com/site/wcpeng/), and conducted in collaboration with Taiwanese professional badminton players. For other official research implementations, please visit [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects/tree/main).
- In recent years, the sports industry has witnessed a significant rise in interest in leveraging artificial intelligence to enhance players' performance. However, the application of deep learning to improve badminton athletes' performance faces challenges related to identifying weaknesses, generating winning suggestions, and validating strategy effectiveness. These challenges arise due to the limited availability of realistic environments and agents. This paper aims to address these research gaps and make contributions to the badminton community. To achieve this goal, we propose a closed-loop approach consisting of six key components: Badminton Data Acquisition, Imitating Players' Styles, Simulating Matches, Optimizing Strategies, Training Execution, and Real-World Competitions.
![Close Look AI Approach](/assets/Close%20Look%20AI%20Approach.png)

## Demo
- Interface link: https://badminton.andyjjrt.cc/
- We also provide demonstrations on how to use these data analysis functions for in-depth analysis and interpretation. ([Evaluation Modules Demonstration](/Evaluation_Modules_Demo/README.md))

## Overview

> This project is an improved version of a [previous one](https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI%20Badminton%20Environment), enhancing the performance of the built-in virtual player, adding rule-based and player movement constraints, and proposing more effective evaluation modules to help players improve their performance. Additionally, we created a website to make the system more accessible to players.

- The CoachAI+ Environment is an interactive environment specifically designed for single-player badminton matches, integrating real-world badminton rules and realistic opponent AIs. The environment offers the following features:
    - Provides a state-of-the-art model for simulating badminton behavior: [RallyNetv2](/RallyNetv2/README.md).
    - Includes a simple behavior cloning model for users to develop and test.
    - Action constraints are based on the largest real-world badminton dataset proposed by our team, [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet).
    - Features a front-end interface connected to a back-end API for match data analysis, with three main components:
        - Shot Evaluation
        - Tactic Evaluation
        - Visualization & Winning/Losing Reason Statistics
    

## 0. Environment Definition
- The environment defines a Markov Decision Process, which includes the following:
    1. Agents: The environment has two player agents, A and B.
    2. States and Actions:
        - `state`  -  (B's shot type, A's position, B's position, shuttle's position)
        - `action`  - (A's shot type, A's shot position, shuttle landing position, A's recovery position, probability distribution of A's shot types)
        - `state` and `action` formats:
            - Position/Landing point: Coordinates (float x, float y)
            - Shot type: Type (int), 10 types in total
            - Shot type probability distribution: (0.2, 0.15, 0.1, ..., 0.2), 10 probabilities in total
                - Shot types include: short service, long service, net shot, clear, push/rush, smash, defensive shot, drive, lob, and drop
    3. Environment coordinate system (as shown in the figure below):
        - The **hitting player** is always the reference point.
        - The center of the court has coordinates (0, 0), and the entire court (including out-of-bounds) is 960 units long and 355 units wide.
        - y-axis: The **hitting player** is always positioned negatively, with the half-court range (-480, 0) and the in-bound range (0, -330). Conversely, the **opponent** is positioned positively, with the half-court range (0, 480) and the in-bound range (0, 330).
        - x-axis: The right side from the **hitting player's** perspective is positive, while the left side is negative. The half-court range is (-177.5, 177.5), and the in-bound range is (-127.5, 127.5). This applies equally to the opponent's perspective.

    4. Transition
        - Badminton is a turn-based sport. The transition is determined by the opponent and the environment's constraints (for a specific agent).
    5. Reward Function
        - The agent scores when the opponent's shot lands out of bounds or the opponent fails to return the shuttle over the net (users can customize this).

    ![Environment Coordinate System](/assets/Environment%20Coordinate%20System.png)


## 1. Environment Structure

The environment consists of two parts: the **BadmintonEnv**, which is used for interaction and training, and the evaluation components for match analysis and visualization (`Shot_Evaluation`, `State_Visualization`, `Tactic_Evaluation`).

### BadmintonEnv

- `Utils` includes:
    - `Constraint.py`: Action constraints are calculated using conditional probabilities based on the ShuttleSet dataset. The pre-processed dataset is located in `input_data`. For pre-processing details, refer to [RallyNetv2](/RallyNetv2/README.md).
        - We design three action constraints based on the correlation between states and actions:
        - ![Action Constraints](/assets/Action%20Constraints.png)
            - **Shot constraint**: Calculates the range of shot types available based on the opponent's shot type and shuttle position (e.g., if the opponent uses a smash, it's unlikely the player will respond with a smash).
                - Cramer's V between (State 0 + State 3) and Action 0: 0.61
            - **Landing constraint**: Calculates the range of landing points based on the player's shot type and position (e.g., a player cannot land a net shot after performing a smash).
                - Cramer's V between (Action 0 + State 1) and Action 2: 0.31
            - **Movement constraint**: Calculates the range of possible receiving positions based on the player's position (i.e., the player's movement distance has limits).
                - Cramer's V between State 1 and Action 1: 0.24
    - `RewardFunction.py`: Reward function for reinforcement learning training (users can define their own reward function).
    - `BaseServe.py`: If the agent lacks a custom serve positioning function, this function samples a suitable position based on the current score from the `input_data` and provides it to the agent as the initial state.
- `Agent`
    - Provides two built-in agent models. Refer to [RallyNetv2](/RallyNetv2/README.md) for training instructions:
        1. Behavior Cloning Agent
        2. RallyNetv2
- `RunEnv.py`: Main program for interacting with the environment.
- `Environment.py`: Functions for the badminton environment.

### Shot_Evaluation, State_Visualize, Tactic_Evaluation

Each folder contains:
- `Utils`: Includes functions for visualization and data pre-processing.
- `main.py`: The main program for execution.

## 2. Build Environment

```bash
conda env create -f environment.yaml
```

## 3. How to Run

### Run Environment

- Both the player and opponent agents can be freely selected, and each must implement two functions:
    - `serve_state()`: Determines the receiving position when not serving.
        - If an agent lacks this function, the built-in function in `BaseServe.py` will be used.
    - `action(state, info, launch)`: Inputs the state (including serve state) and outputs the action.

- Usage flow:
    - Modify `RunEnv.py` as needed.
    - Decide which agent to use for both players (e.g., BC Agent, RallyNet...).
    - Parameter settings:
        - `player_agent`: Your agent.
        - `opponent_agent`: Opponent's agent.
        - `episodes`: Number of training episodes (number of rallies).
        - `is_match`: Whether to output results according to badminton match rules.
        - `is_constraint`: Whether to enable environment constraints to prevent unrealistic actions.
        - `is_have_serve_state`: Whether your agent has the `BaseServe` function.
        - `filepath`: Output file path.
    - Once the settings are complete, run the program, and it will simulate the match and output the results to `output_game.csv`.

    ```python
    from Environment import BadmintonEnv
    from Agent.BadmintonAgent import BadmintonAgent

    with open('./Datapreprocessing/target_players_ids.pkl', 'rb') as f:
        target_players = pickle.load(f) # List of 27 real player names from ShuttleSet
    
    player_name = 'CHOU Tien Chen'  # You can also create custom names
    opponent_name = 'Kento MOMOTA'  # Choose from real player names

    player_agent = BadmintonAgent(player_name)  # Use any agent
    opponent_agent = BadmintonAgent(opponent_name)  # Use any agent
    rally_length = 50

    env = BadmintonEnv(player_agent, opponent_agent, player_name, opponent_name, episodes, 
                       is_match=True, is_constraint=True, is_have_serve_state=True, filepath='./output_data/output_game.csv')

    for rally in range(1, rally_length + 1):
        states, info, done, launch = env.reset()
        while not done:
            action = player_agent.action(states, launch)
            states, reward, info, done, launch = env.step(action, launch)
    env.close()
    ```

### Run Evaluation ([Demo](/Evaluation_Modules_Demo/README.md))

- To analyze real or environment-generated data, three folders correspond to different evaluations:
    - **Shot Evaluation**: 
        - Rally Length vs. Energy Expenditure Diagram
        - Shot Type Visualization
        - Rally Shot-by-Shot Momentum and Pace Plot
    - **Tactic Evaluation**: Shows tactical usage and win rate correlations, as well as distribution graphs (pie charts and bar charts).
    - **Visualization & Winning/Losing Reason Statistics**: Provides statistics on the top four winning/losing states and actions, along with accumulated shot type distribution and rally animations.

- Usage flow:
    - Open `main.py` in the respective folders.
    - Set the path to the **real match data**.
        - Match data format is described in the Data Preprocessing section of [RallyNetv2](/RallyNetv2/README.md).
    - Set player names and other parameters.


### Run Web Page ([Demo](https://badminton.andyjjrt.cc))

- To run web page, simply run following script in project's root folder.
    ```
    cd frontend
    pnpm install
    pnpm run build
    cp dist/index.html dist/404.html
    cd ..
    uvicorn api.app:app --host 0.0.0.0 --port 8000
    ```
- You can refer to [frontend](/frontend/README.md) and [api](/api/README.md) to learn more about setting up the website.

## Reference
[1] Wang et al. "Offline Imitation of Badminton Player Behavior via Experiential Contexts and Brownian Motion." ECML PKDD'24.  
[2] Wang et al. "How Is the Stroke? Inferring Shot Influence in Badminton Matches via Long Short-term Dependencies." ACM TIST'24  
[3] Wang et al. "ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis." KDD'23  
[4] Wang et al. "The CoachAI Badminton Environment: A Novel Reinforcement Learning Environment with Realistic Opponents (Student Abstract)." AAAI'24.  
[5] Wang et al. "The CoachAI Badminton Environment: Bridging the Gap between a Reinforcement Learning Environment and Real-World Badminton Games." AAAI'24  
[6] Wang, Kuang-Da. "Enhancing Badminton Player Performance via a Closed-Loop AI Approach: Imitation, Simulation, Optimization, and Execution." CIKM'24  

