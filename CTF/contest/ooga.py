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
import random, util
from game import Directions
from baselineTeam import ReflexCaptureAgent
import math

#################
# Team creation #
#################
import util
from game import Actions


class Inference():

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, gameState, index):
        self.pos = None
        self.index = index
        self.initialize(gameState)
        self.distancer = None

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeStarting(gameState)

    def initializeStarting(self, gameState):
        location1 = gameState.getInitialAgentPosition(self.index)
        location2 = gameState.getInitialAgentPosition((self.index + 2) % 4)
        self.belief = util.Counter()
        self.belief[location1] = 0.5
        self.belief[location2] = 0.5
        self.pacman = False

    # when agent turns from pacman to ghost and when we are unsure it was capture or not
    def halfReinitialize(self, gameState):
        location = gameState.getInitialAgentPosition(self.index)
        self.belief.normalize()
        self.belief[location] += 1
        self.belief.normalize()
        self.pacman = gameState.getAgentState(self.index).isPacman

    def createTransitionModel(self, gameState, noisyDistance, previousBelief, pacmanPosition):
        transitionModel = util.Counter()
        actions = Actions()
        # 1 if pacman else -1
        pacman = 1 if gameState.getAgentState(self.index).isPacman else -1
        # 1 if enemy is on the left side else -1
        reverseBoard = 1 if gameState.getInitialAgentPosition(self.index)[0] < gameState.data.layout.width // 2 else -1

        distanceToFood = {}
        ourFoods = gameState.getBlueFood().asList() if gameState.isOnRedTeam(
            self.index) else gameState.getRedFood().asList()
        for pos in previousBelief:
            neighbors = actions.getLegalNeighbors(pos, gameState.data.layout.walls)
            # leave only the neighbors that are within distance 6 of noisy reading
            # and on the correct side (consider both whether agent is pacman or not and whether enemy is on left or right)

            # potentially useful
            neighbors = filter(lambda x: abs(
                util.manhattanDistance(x, pacmanPosition) - noisyDistance) <= 7
                                         and x[0] * pacman * reverseBoard >= gameState.data.layout.width // 2 * pacman *
                                         reverseBoard and util.manhattanDistance(x, pacmanPosition) > 4, neighbors)
            if pos not in distanceToFood:
                distanceToFood[pos] = min([self.distancer.getDistance(food, pos) for food in ourFoods])
            if not neighbors:
                continue
            for nei in neighbors:
                if nei not in distanceToFood:
                    distanceToFood[nei] = min([self.distancer.getDistance(food, nei) for food in ourFoods])
            # work under assumption that enemy ghosts are more likely to move than stay
            if neighbors == [pos]:
                transitionModel[pos] += previousBelief[pos] * 0.2 / (
                        abs(util.manhattanDistance(pos, pacmanPosition) - noisyDistance) + 1)
                continue
            probStay = 0.2 * previousBelief[pos]
            if pos in neighbors:
                probNei = previousBelief[pos] / (len(neighbors) - 1) * 0.8
            else:
                probNei = previousBelief[pos] / (len(neighbors))
            for nei in neighbors:
                prob = probStay if pos == nei else probNei
                dist = util.manhattanDistance(nei, pacmanPosition)
                foodWeight = (distanceToFood[pos] - distanceToFood[nei]) * 0.2 + 1
                transitionModel[nei] += prob / (abs(dist - noisyDistance) + 1) * foodWeight

        if not transitionModel:
            self.initializeStarting(gameState)
            transitionModel = self.belief
        transitionModel.normalize()
        return transitionModel

    def observe(self, observation, agentPos, gameState):
        # if observation is a tuple, we know exact location
        pacmanAgain = gameState.getAgentState(self.index).isPacman
        if isinstance(observation, tuple):
            self.belief = util.Counter()
            self.belief[observation] = 1.0
            self.pacman = pacmanAgain
            return

        noisyDistance = observation
        pacmanPosition = agentPos

        if self.pacman and not pacmanAgain:
            startingLocation = gameState.getInitialAgentPosition(self.index)
            if abs(util.manhattanDistance(startingLocation, pacmanPosition) - noisyDistance) <= 6:
                self.halfReinitialize(gameState)
                return

        beliefs = self.getBeliefDistribution()
        probs = self.createTransitionModel(gameState, noisyDistance, beliefs, pacmanPosition)

        self.belief = probs
        self.pacman = pacmanAgain

    def getBeliefDistribution(self):
        self.belief = self.deleteNullValues(self.belief)
        return self.belief

    def getMostLikelyPosition(self):
        if self.pos:
            return self.pos
        beliefs = self.getBeliefDistribution()
        mostLikelyPos = max(beliefs.keys(), key=lambda x: beliefs[x])
        return mostLikelyPos

    def getTopFiveMostLikelyPosition(self):
        beliefs = self.getBeliefDistribution()
        beliefs_print = [(coord, beliefs[coord]) for coord in beliefs]
        beliefs_print.sort(reverse=True, key=lambda x: x[1])
        return beliefs_print[:5]

    def deleteNullValues(self, counter):
        newCounter = util.Counter()
        for c in counter:
            if counter[c] != 0:
                newCounter[c] = counter[c]
        return newCounter


Inferences = {}


def createTeam(firstIndex, secondIndex, isRed,
               first='GenericReflexAgent', second='GenericReflexAgent'):
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
    return [eval(first)(firstIndex, "Offensive"), eval(second)(secondIndex, "Defensive")]


class GenericReflexAgent(ReflexCaptureAgent):

    def __init__(self, index, offOrDef):
        ReflexCaptureAgent.__init__(self, index)
        if offOrDef == "Offensive":
            self.position = "Offensive"

        else:
            self.position = "Defensive"

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """
        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        '''
        Your initialization code goes here, if you need any.
        '''

        self.come_back = False
        index = 0 if self.index % 2 == 1 else 1
        if not Inferences:
            Inferences[index] = Inference(gameState, index)
            Inferences[index + 2] = Inference(gameState, index + 2)
        for key in Inferences:
            Inferences[key].initialize(gameState)
            Inferences[key].distancer = self.distancer

    ### DEFENSIVE METHODS
    def defChooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        max = -999999

        max_action = None

        opponent_indices = self.getOpponents(gameState)
        invaders = {index: gameState.getAgentState(index).isPacman for index in opponent_indices}

        myCurrFood = self.getFoodYouAreDefending(gameState).asList()
        myLastFood = self.getFoodYouAreDefending(
            self.getPreviousObservation()).asList() if self.getPreviousObservation() != None else myCurrFood
        recentlyEatenFood = list(set(myCurrFood) ^ set(myLastFood))

        # pass in observations, update particles
        for enemy in self.getOpponents(gameState):
            # observation is either exact position if within detection range, or an approximate distance
            enemyPosition = gameState.getAgentPosition(enemy)
            observation = enemyPosition if enemyPosition else gameState.getAgentDistances()[enemy]
            Inferences[enemy].observe(observation, myPos, gameState)
            otherEnemy = (enemy + 2) % 4
            if len(recentlyEatenFood) == 1 and gameState.getAgentState(
                    enemy).isPacman and (
                    (recentlyEatenFood[0] in Inferences[enemy].belief.keys() and recentlyEatenFood[0] not in Inferences[
                        otherEnemy].belief.keys()) or not gameState.getAgentState(
                otherEnemy).isPacman):
                Inferences[enemy].observe(recentlyEatenFood[0], myPos, gameState)

        # consider all actions and calculate their features
        for action in actions:
            features = self.getFeatures(gameState, action)
            weights = self.defGetWeights(features['numInvaders'])
            potential_max = features * weights
            if potential_max > max:
                max = potential_max
                max_action = action
        return max_action

    def defGetFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        myPos = (int(myPos[0]), int(myPos[1]))

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # figure out which enemies are pacmen - if neither, then consider both enemies
        opponent_indices = self.getOpponents(successor)
        invaders = {index: successor.getAgentState(index).isPacman for index in opponent_indices}

        to_attack = [index for index in opponent_indices if
                     invaders[index]] if True in invaders.values() else opponent_indices

        enemyPositions = [Inferences[enemy].getMostLikelyPosition() for enemy in to_attack]
        enemyDistances = [self.distancer.getDistance(myPos, enemyPos) for enemyPos in enemyPositions]

        self.displayDistributionsOverPositions([inference.belief for inference in Inferences.values()])

        features['numInvaders'] = len([a for a in invaders.values() if a == True])
        features['invaderDistance'] = min(enemyDistances)
        shouldRunAway = -1 if gameState.getAgentState(self.index).scaredTimer >= features['invaderDistance'] // 2 and \
                              features['invaderDistance'] <= 2 else 1
        if shouldRunAway < 0:
            print("SHOULD RUNNNN")
        features['invaderDistance'] *= shouldRunAway

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def defGetWeights(self, numInvaders):
        onDefense = 0
        if numInvaders > 0:
            onDefense = 100
        return {'numInvaders': -1000, 'onDefense': onDefense, 'invaderDistance': -3, 'stop': -1, 'reverse': -1}

    ### OFFENSIVE METHODS

    def forcedDeadEnd(self, action, gameState, visited, current_depth, max_depth):

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        feasible_actions = [action for action in successor.getLegalActions(self.index) if myPos not in visited]
        if current_depth == max_depth:
            return False
        elif len(feasible_actions) == 0:
            return True
        else:
            visited.append(myPos)
            if False in [self.forcedDeadEnd(action, successor, visited, current_depth + 1, max_depth) for action in
                         feasible_actions]:
                return False
            else:
                return True

    def offChooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        '''onBorderGhosts = 
        if features['closest_ghost'] < 6 and not gameState.getAgentState(self.index).isPacman:
            moveVert = True
        else:
            moveVert = False'''

        max = -999999
        safe_max = -999999
        vert_max = -999999
        max_action = None

        # when agent has eaten at least half of the remaining food (round half up), come back to own side
        if gameState.getAgentState(self.index).numCarrying > (math.ceil(len(self.getFood(gameState).asList()) / 2) - 1) \
                and gameState.getAgentState(self.index).isPacman:
            self.come_back = True

        # return to offense when pellets are returned safely
        else:
            if not gameState.getAgentState(self.index).isPacman:
                self.come_back = False

        myCurrFood = self.getFoodYouAreDefending(gameState).asList()
        myLastFood = self.getFoodYouAreDefending(
            self.getPreviousObservation()).asList() if self.getPreviousObservation() != None else myCurrFood
        recentlyEatenFood = list(set(myCurrFood) ^ set(myLastFood))

        # pass in observations, update particles
        for enemy in self.getOpponents(gameState):
            # observation is either exact position if within detection range, or an approximate distance
            enemyPosition = gameState.getAgentPosition(enemy)
            observation = enemyPosition if enemyPosition else gameState.getAgentDistances()[enemy]
            Inferences[enemy].observe(observation, myPos, gameState)
            otherEnemy = (enemy + 2) % 4
            if len(recentlyEatenFood) == 1 and gameState.getAgentState(
                    enemy).isPacman and (
                    recentlyEatenFood[0] in Inferences[enemy].belief.keys() or not gameState.getAgentState(
                otherEnemy).isPacman):
                Inferences[enemy].observe(recentlyEatenFood[0], myPos, gameState)
        self.displayDistributionsOverPositions([inference.belief for inference in Inferences.values()])
        # consider all actions and calculate their features
        for action in actions:
            features = self.getFeatures(gameState, action)

            weights = self.offGetWeights(features['closest_ghost'])
            potential_max = features * weights
            '''print("Q VALUE: " + str(potential_max))
            if features['moveVertically'] == 1:
                if potential_max >'''
            if features[
                'closest_ghost'] > 5:  # if an action is "safe", i.e. takes you outside of ghost's striking distance
                # if features['moveVertically'] == 1:
                if potential_max > safe_max:  # update max safe action if the associated Q value is the new highest
                    safe_max = potential_max
                    max_action = action
            else:
                if not self.forcedDeadEnd(action, gameState, [myPos], 0,
                                          10):  # if you are being pursued, prune actions that lead to a forced dead end
                    print("safe")
                    if safe_max == -999999:  # if no safe actions have been seen yet in the current loop
                        if potential_max > max:  # update max action if the associated Q value is the new highest
                            max = potential_max
                            max_action = action
                else:
                    print("dead end")

        if max_action == None:
            return random.choice(actions)
        return max_action

    def offGetFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -1 * len(foodList)  # self.getScore(successor)
        myPos = successor.getAgentState(self.index).getPosition()

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            if not self.come_back:  # if Pacman has the green light to keep eating food
                minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
                features['distanceToFood'] = minDistance
            else:  # if pacman does not have the green light to eat, go back to your own side
                features['distanceToFood'] = min(
                    [self.getMazeDistance(myPos, food) for food in self.getFoodYouAreDefending(gameState).asList()])

        # figure out which enemies are ghosts - if neither, then consider both enemies
        opponent_indices = [a for a in self.getOpponents(successor)]
        enemyPositions = [gameState.getAgentState(index).getPosition()
                          if not gameState.getAgentState(index).isPacman and gameState.getAgentState(
            index).scaredTimer < 5
                          else None for index in opponent_indices]
        enemyDistances = [self.distancer.getDistance(myPos, enemyPos) if enemyPos != None else 30 for enemyPos in
                          enemyPositions]
        features['closest_ghost'] = min(enemyDistances)

        return features

    def offGetWeights(self, closest_ghost):
        if closest_ghost < 6:
            return {'successorScore': 0, 'distanceToFood': 0, 'closest_ghost': 1000, 'moveVertically': 0}
        else:
            return {'successorScore': 1000, 'distanceToFood': -20, 'closest_ghost': 0, 'moveVertically': 0}

    def getFeatures(self, gameState, action):
        if self.position == "Offensive":
            return self.offGetFeatures(gameState, action)
        else:
            return self.defGetFeatures(gameState, action)

    def chooseAction(self, gameState):
        if self.getPreviousObservation() != None:
            myDistanceChange = self.distancer.getDistance(gameState.getAgentState(self.index).getPosition(),
                                                          self.getPreviousObservation().getAgentState(
                                                              self.index).getPosition())

            teammateIndex = [i for i in self.getTeam(gameState) if i != self.index][0]
            teammateDistanceChange = self.distancer.getDistance(gameState.getAgentState(teammateIndex).getPosition(),
                                                                self.getPreviousObservation().getAgentState(
                                                                    teammateIndex).getPosition())

            if self.position == "Offensive":
                if myDistanceChange > 5:
                    self.position = "Defensive"
            else:
                if teammateDistanceChange > 5:
                    self.position = "Offensive"

        if self.position == "Offensive":
            return self.offChooseAction(gameState)
        else:
            return self.defChooseAction(gameState)