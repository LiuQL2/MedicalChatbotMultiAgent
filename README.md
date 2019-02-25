This is the code of a task-oriented dialogue system for automatic diagnosis, where a hierarchical reinforcement learning
are implemented as the dialogue policy. The low level RL consists of multi-agents and each agent is the specific policy
in terms of a type of disease. While the high level RL is responsible for selecting one of the low level agent and the 
selected agent will interact with patient in this dialogue turn.

Under working...

At this time, AgentDQN is the flat-DQN model, AgentWIthGoal2 and AgentWithGoal3 are the HRL-based agents. And the difference
is written in the corresponding py files. And the other agents can be ignored.