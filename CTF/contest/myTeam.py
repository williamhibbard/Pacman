# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffenseAgent', second = 'FlexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
              
class OffenseAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.timeLeft = 1200

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    isRed = gameState.isOnRedTeam(self.index)
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    pos = successor.getAgentState(self.index).getPosition()
    opponents = self.getOpponents(successor)
    homeFoodList = self.getFoodYouAreDefending(gameState).asList()
    if isRed:
      capsules = gameState.getBlueCapsules()
      foodList = gameState.getBlueFood().asList()
    else:
      capsules = gameState.getRedCapsules()
      foodList = gameState.getRedFood().asList()

    features['successorScore'] = self.getScore(successor)   
    
    "min distance back to base (own food)"
    minDistHome = float("inf")
    for food in homeFoodList: 
      distance = self.getMazeDistance(pos, food)
      if distance < minDistHome: 
        minDistHome = distance
    features['minDistHome'] = minDistHome

    "if capture opponent"
    features['captureOpponent'] = 0.0
    for opponent in opponents:
      opponentPos = successor.getAgentPosition(opponent)
      if pos == opponentPos:
        features['captureOpponent'] = 1.0

    "min distance to food"
    minFoodDist = float("inf")
    if pos in foodList: 
      features['foodCaptured'] = 1.0
      minFoodDist = 0.0
    else:
      features['foodCaptured'] = 0.0
      for food in foodList:
        distance = self.getMazeDistance(pos, food)
        if distance < minFoodDist: 
          minFoodDist = distance
          features['minFoodDist'] = minFoodDist
      features['minFoodDist'] = minFoodDist

    "min distance to capsule"
    minCapsuleDist = float("inf")
    if capsules: 
      if pos in capsules: 
        features['capsuleCaptured'] = 1.0
        minCapsuleDist = 0
      else: 
        features['capsuleCaptured'] = 0.0
        for capsule in capsules: 
          distance = self.getMazeDistance(pos, capsule)
          if distance < minCapsuleDist: 
            minCapsuleDist = distance
    else: 
      minCapsuleDist = 0.0
    features['minCapsuleDist'] = minCapsuleDist

    "min distance to opponent"
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    features['numDefenders'] = len(defenders)
    if len(defenders) > 0: #observable opponent
      dists = [self.getMazeDistance(pos, a.getPosition()) for a in defenders]
      features['defenderDistance'] = min(dists)
    else: # not obseravable
      features['minDefenderReading'] = min(successor.getAgentDistances())
    if action == Directions.STOP:
      features['stop'] = 1.0

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    for agent in self.getOpponents(gameState):
      scaredTimer = gameState.getAgentState(agent).scaredTimer
      
    if scaredTimer > 5:
      return {'successorScore': 1.0, 'minFoodDist': -1.0, 'foodCaptured': 100.0, 'minCapsuleDist': 0.0, 'capsuleCaptured': 0.0,
      'defenderDistance': -0.0, 'minDefenderReading': 0.0, 'captureOpponent': 1000.0, 'minDistHome': 0.0, 'stop': float('-inf')}
    elif gameState.getAgentState(self.index).numCarrying >= 10: 
      return {'successorScore': 0.0, 'minFoodDist': -0.0, 'foodCaptured': 100.0, 'minCapsuleDist': -0.0, 'capsuleCaptured': 0.0,
      'defenderDistance': 1000.0, 'minDefenderReading': 100.0, 'captureOpponent': 0.0, 'minDistHome': -1.0, 'stop': float('-inf')}
    elif self.timeLeft <= 60:
      return {'successorScore': 0.0, 'minFoodDist': -0.0, 'foodCaptured': 100.0, 'minCapsuleDist': -0.0, 'capsuleCaptured': 0.0,
      'defenderDistance': 1000.0, 'minDefenderReading': 100.0, 'captureOpponent': 0.0, 'minDistHome': -2.0, 'stop': float('-inf')}
    else: 
      return {'successorScore': 1.0, 'minFoodDist': -1.0, 'foodCaptured': 500.0, 'minCapsuleDist': -2.0, 'capsuleCaptured': 1000.0,
      'defenderDistance': 1000.0, 'minDefenderReading': 100.0, 'captureOpponent': 0.0, 'minDistHome': 0.0,'stop': float('-inf')}

  def chooseAction(self, gameState):
    self.timeLeft = self.timeLeft - 1
    actions = gameState.getLegalActions(self.index)
    bestAction = Directions.STOP
    bestScore = float("-inf")

    for action in actions:
      score = self.evaluate(gameState, action)
      if score > bestScore: 
        bestScore = score
        bestAction = action

    #print(self.capsuleTimer)
    return bestAction

class DefenseAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.food = CaptureAgent.getFood(self, gameState)
    self.foodDef = CaptureAgent.getFoodYouAreDefending(self, gameState)
    self.opponents = CaptureAgent.getOpponents(self, gameState)
    self.team = CaptureAgent.getTeam(self, gameState)
    self.capsuleTime = 0

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    if action == Directions.STOP: features['stop'] = 1
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    pos = successor.getAgentState(self.index).getPosition()
    opponents = self.getOpponents(successor)
    capsules = self.getCapsulesYouAreDefending(successor)
    foodList = self.getFoodYouAreDefending(successor).asList()

    "capsule distance"
    minCapsuleDist = float("inf")
    for capsule in capsules: 
      distance = self.getMazeDistance(pos, capsule)
      if distance < minCapsuleDist: 
        minCapsuleDist = distance
    features['minCapsuleDist'] = minCapsuleDist

    "opponent distance"
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(pos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP:
      features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate. They can be either
    a counter or a dictionary.
    """
    "if scared"
    if gameState.getAgentState(self.index).scaredTimer > 0: 
      return {'numInvaders': -1000, "invaderDistance": 10.0, 'minCapsuleDist': -2.0, 'stop': -100.0}
    return {'numInvaders': -1000, "invaderDistance": -10.0, 'minCapsuleDist': -2.0, 'stop': -100.0}

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    bestAction = actions[0]
    bestScore = float("-inf")

    for action in actions:
      score = self.evaluate(gameState, action)
      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction

class FlexAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.food = CaptureAgent.getFood(self, gameState)
    self.foodDef = CaptureAgent.getFoodYouAreDefending(self, gameState)
    self.opponents = CaptureAgent.getOpponents(self, gameState)
    self.team = CaptureAgent.getTeam(self, gameState)
    self.timeLeft = 1200
    self.capsuleTime = 0

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    if action == Directions.STOP: features['stop'] = 1
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    pos = successor.getAgentState(self.index).getPosition()
    opponents = self.getOpponents(successor)
    capsules = self.getCapsulesYouAreDefending(successor)
    foodList = self.getFoodYouAreDefending(successor).asList()
    isRed = gameState.isOnRedTeam(self.index)

    if isRed:
      enemyFoodList = gameState.getBlueFood().asList()
    else:
      enemyFoodList = gameState.getRedFood().asList()

    "capsule distance"
    minCapsuleDist = float("inf")
    for capsule in capsules: 
      distance = self.getMazeDistance(pos, capsule)
      if distance < minCapsuleDist: 
        minCapsuleDist = distance
    if minCapsuleDist > 1000.0:
      minCapsuleDist = 0.0
    features['minCapsuleDist'] = minCapsuleDist

    "opponent distance"
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0: # if observable
      dists = [self.getMazeDistance(pos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    else: # not observable
      features['minInvaderReading'] = min(successor.getAgentDistances())

    "min distance to enemy food"
    minFoodDist = float("inf") 
    if pos in enemyFoodList: 
      features['foodCaptured'] = 1.0
      minFoodDist = 0.0
    else:
      features['foodCaptured'] = 0.0
      for food in enemyFoodList: 
        distance = self.getMazeDistance(pos, food)
        if distance < minFoodDist: 
          minFoodDist = distance
    features['minFoodDist'] = minFoodDist

    "average distance to team food"
    count = 0
    totalDistance = 0
    features['foodDist'] = 0.0
    for food in foodList:
      count += 1
      totalDistance += self.getMazeDistance(pos, food)
    features['foodDist'] = float(totalDistance) / float(count)

    if action == Directions.STOP:
      features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate. They can be either
    a counter or a dictionary.
    """
    for agent in self.getOpponents(gameState):
      scaredTimer = gameState.getAgentState(agent).scaredTimer

    risky = False
    if self.getScore(gameState) < 0 and self.timeLeft <= 100:
      risky = True
      
    if scaredTimer > 5 or risky:
      return {'numInvaders': 0.0, "invaderDistance": 0.0, 'minInvaderReading': 0.0, 'minCapsuleDist': 0.0, 'stop': -100.0, 'foodCaptured': 1000.0, 'minFoodDist': -1.0, 'foodDist': 0.0}
    elif gameState.getAgentState(self.index).scaredTimer > 0: #scared
      return {'numInvaders': -0.0, "invaderDistance": 1000.0, 'minInvaderReading': 10.0, 'minCapsuleDist': -0.0, 'stop': -100.0, 'foodCaptured': 0.0, 'minFoodDist': 0.0, 'foodDist': 0.0}
    else: 
      return {'numInvaders': -1000, "invaderDistance": -1000.0, 'minInvaderReading': -100.0, 'minCapsuleDist': -1.0, 'stop': -100.0, 'foodCaptured': 0.0, 'minFoodDist': 0.0, 'foodDist': -1.0}

  def chooseAction(self, gameState):
    self.timeLeft -= 1
    actions = gameState.getLegalActions(self.index)
    bestAction = actions[0]
    bestScore = float("-inf")

    for action in actions:
      score = self.evaluate(gameState, action)
      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction


