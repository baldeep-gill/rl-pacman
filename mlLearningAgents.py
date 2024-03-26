# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

import numpy as np

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.legal = state.getLegalPacmanActions()

        # For use with __hash__ and __eq__
        self.state = state
        self.position = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood()

    # Hash and Equals functions inspired by the definitions in GameState and GameStateData
    def __hash__(self):
        return int((hash(self.position) + 9 * hash(self.food) + 13 * hash(str(self.ghostPositions))) % 1048575)

    def __eq__(self, other):
        if other is None:
            return False
        if not self.state.getPacmanState() == other.state.getPacmanState():
            return False
        if not self.state.getGhostStates() == other.state.getGhostStates():
            return False
        if not self.food == other.food:
            return False
        
        return True


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.13,
                 epsilon: float = 0.1,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()

        # Both are dictionaries of dictionaries. State will be used as the first key and an action will be used as the second key
        self.qTable = {} # Q-table holding Q-values for each state/action pair
        self.nTable = {} # Occurence table holding the number of times each state/action pair occurs

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """

        return (endState.getScore() - startState.getScore())

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """

        # If we have not seen the state before, put it in the Q-table initialising it with an empty dict
        if state not in self.qTable:
            self.qTable.update({state: {}})

        # If we have not seen this action before for the given state, initialise this state/action pair with 0
        if action not in self.qTable.get(state):
            self.qTable.get(state).update({action: float(0)})

        return self.qTable.get(state).get(action)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        vals = []

        # Get Q-values for each of the possible actions in this state
        for action in state.legal:
            vals.append(self.getQValue(state, action))

        # Return the largest Q-value or 0 if there aren't any
        return float(0) if len(vals) == 0 else max(vals)

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        
        # Get Q[s, a]
        qVal = self.getQValue(state, action)
        # Get max{a'} Q[s', a']
        max_val = self.maxQValue(nextState)
        
        # Perform update rule
        update_value = qVal + (self.alpha * (reward + (self.gamma * max_val) - qVal))
        
        # update Q-table with new Q-value
        self.qTable.get(state).update({action: update_value})

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # If we have not seen the state before, put it in the count-table
        if state not in self.nTable:
            self.nTable.update({state: {}})

        # If we have not seen the action before, put it in the count-table with a value of 0
        if action not in self.nTable.get(state):
            self.nTable.get(state).update({action: 0})

        # Increment the count for this state/action pair
        # Unseen state/action pairs will get correctly incremented to 1 due to setting them to 0 in the check
        self.nTable.get(state).update({action: self.nTable.get(state).get(action) + 1})

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        # If we have not seen the state before, put it in the count-table
        if state not in self.nTable:
            self.nTable.update({state: {}})

        # If we have not seen the action before, put it in the count-table with a value of 0
        if action not in self.nTable.get(state):
            self.nTable.get(state).update({action: 0})

        return self.nTable.get(state).get(action)

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        optimistic_reward = 0.04
        N = 35

        return optimistic_reward + utility if counts < N else utility

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state) # Current state
        vals = [] # Temp list to hold Q-values
        explore = util.flipCoin(self.epsilon) # Set up e-greedy exploration

        for i in range(0, len(legal)):
            # vals.append(self.getQValue(stateFeatures, legal[i]))

            # For each legal action, get the Q-value and the count for the state/action pair,
            # and pass these values to the exploration function.
            vals.append(self.explorationFn(self.getQValue(stateFeatures, legal[i]), self.getCount(stateFeatures, legal[i])))

        # Choose if we do an action through e-greedy
        if explore: 
            action = random.choice(legal)
        else:
            # Get the action with the largest Q-value after running exploration function
            action = legal[np.argmax(vals)] 
        
        self.updateCount(stateFeatures, action)
        next_state = state.generatePacmanSuccessor(action)

        self.learn(stateFeatures, action, self.computeReward(state, next_state), GameStateFeatures(next_state))
        # bosh
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            # print(self.qTable)
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
