# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        """
        print("current state: ", gameState.getPacmanPosition())
        print("chosen move: ", legalMoves[chosenIndex])
        """
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.0

        if action == "Stop":
            score -= 100.0

        "Food"

        "If new position has food"
        currentFood = currentGameState.getFood()
        if currentFood[newPos[0]][newPos[1]]:
            score += 5.0

        count = 0
        totalDistance = 0.0
        closestFoodDistance = float("-inf")
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    count += 1
                    distance = manhattanDistance(newPos, (i, j))
                    totalDistance += distance
                    if distance < closestFoodDistance:
                        closestFoodDistance = distance
        if count == 0:
            score = float("inf")
        else:
            avgDistance = totalDistance / count
            score += 1.0 / avgDistance
            score += 5.0 / closestFoodDistance

        "Ghosts"
        avgGhostDistance = 0.0
        for newGhostState in newGhostStates:
            distance = manhattanDistance(newPos, newGhostState.getPosition())
            avgGhostDistance += distance
            "If not scared"
            if newGhostState.scaredTimer == 0:
                "If ghost is close"
                if distance <= 2:
                    score -= 10.0
            "If scared"
            if newGhostState.scaredTimer != 0:
                if distance == 0:
                    score += 1.0

        "Capsules"
        for capsule in successorGameState.getCapsules():
            if newPos == capsule:
                score += 1.0

        """
        print("action: ", action, " score: ", score)
        """
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def max_fxn(state, depth):

            "terminal"
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            bestAction = None
            bestScore = float("-inf")
            score = bestScore
            pacmanActions = state.getLegalActions(0)

            for action in pacmanActions:
                #print("pacman action: ", action)
                score = min_fxn(state.generateSuccessor(0, action), 1, depth)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            if depth != self.depth:
                return bestScore
            else:
                return bestAction

        def min_fxn(state, agent, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            worstScore = float("inf")
            score = worstScore
            ghostActions = state.getLegalActions(agent)
            #if last ghost
            if agent == state.getNumAgents() - 1: 
                for action in ghostActions:
                    score = max_fxn(state.generateSuccessor(agent, action), depth - 1)
                    if score < worstScore: 
                        worstScore = score
            #if not last ghost
            else: 
                for action in ghostActions: 
                    score = min_fxn(state.generateSuccessor(agent, action), agent + 1, depth)
                    if score < worstScore: 
                        worstScore = score
            return worstScore

        return max_fxn(gameState, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_fxn(state, depth, alpha, beta):

            "terminal"
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            bestAction = None
            bestScore = float("-inf")
            score = bestScore
            pacmanActions = state.getLegalActions(0)

            for action in pacmanActions:
                # print("pacman action: ", action)
                score = min_fxn(state.generateSuccessor(0, action), 1, depth, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                if bestScore > beta:
                    return bestScore
                alpha = max(bestScore, alpha)
            if depth != self.depth:
                return bestScore
            else:
                return bestAction

        def min_fxn(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            worstScore = float("inf")
            score = worstScore
            ghostActions = state.getLegalActions(agent)
            # if last ghost
            if agent == state.getNumAgents() - 1:
                for action in ghostActions:
                    score = max_fxn(state.generateSuccessor(agent, action), depth - 1, alpha, beta)
                    if score < worstScore:
                        worstScore = score
                    if worstScore < alpha:
                        return worstScore
                    beta = min(worstScore, beta)
            # if not last ghost
            else:
                for action in ghostActions:
                    score = min_fxn(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta)
                    if score < worstScore:
                        worstScore = score
                    if worstScore < alpha:
                        return worstScore
                    beta = min(worstScore, beta)
            return worstScore

        return max_fxn(gameState, self.depth, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_fxn(state, depth):

            "terminal"
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            bestAction = None
            bestScore = float("-inf")
            score = bestScore
            pacmanActions = state.getLegalActions(0)

            for action in pacmanActions:
                #print("pacman action: ", action)
                score = min_fxn(state.generateSuccessor(0, action), 1, depth)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            if depth != self.depth:
                return bestScore
            else:
                return bestAction

        def min_fxn(state, agent, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            totalScore = 0.0
            ghostActions = state.getLegalActions(agent)
            #if last ghost
            if agent == state.getNumAgents() - 1: 
                for action in ghostActions:
                    totalScore += max_fxn(state.generateSuccessor(agent, action), depth - 1)
            #if not last ghost
            else: 
                for action in ghostActions: 
                    totalScore += min_fxn(state.generateSuccessor(agent, action), agent + 1, depth)
            return totalScore / len(ghostActions)

        return max_fxn(gameState, self.depth)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We considered the three most important things: food, capsules, and ghosts.
      The more food there is on the board, the lower the score.  We also lower your score if your average distance
      from food pellets is too high.
      Next are ghosts.  Simply put, being far away from angry ghosts is good, and you get deducted more the closer you
      are.  When ghosts are scared, you are rewarded for getting closer to the point of eating them.
      Lastly, for every pellet left on the board, you are deducted points.  You are rewarded for being next to them, so
      there is incentive to find them and eat them, and then at that point there is incentive to eat the ghosts.
    """
    "*** YOUR CODE HERE ***"
    score = 0.0
    pacman = currentGameState.getPacmanState()
    ghosts = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    pos = pacman.getPosition()

    "Food"

    count = 0
    totalDistance = 0.0
    closestFoodDistance = float("inf")
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                count += 1
                distance = manhattanDistance(pos, (i, j))
                totalDistance += distance
                if distance < closestFoodDistance:
                    closestFoodDistance = distance
    if count == 0:
        score = 9999.0
    else:
        avgDistance = totalDistance / count
        score += 2.0 / avgDistance
        #score += 5.0 / closestFoodDistance
        score -= float(count)


    "Ghosts"
    avgGhostDistance = 0.0
    for ghost in ghosts:
        distance = manhattanDistance(pos, ghost.getPosition())
        avgGhostDistance += distance
        if ghost.scaredTimer == 0:
            if distance <= 2:
                score -= 10.0
            else:
                score += distance * 0.01
        else:
            score -= distance * 1.0
        "If not scared"
        """
        if ghost.scaredTimer == 0:
            "If ghost is close"
            if distance <= 2:
                score -= 10.0
        "If scared"
        if ghost.scaredTimer != 0:
            if distance == 0:
                score += 20.0
        """
    "Capsules"
    for capsule in currentGameState.getCapsules():

        score -= 12.0
        if (pos[0] + 1, pos[1]) == capsule:
            score += 1.0
        if (pos[0] - 1, pos[1]) == capsule:
            score += 1.0
        if (pos[0], pos[1] + 1) == capsule:
            score += 1.0
        if (pos[0], pos[1] - 1) == capsule:
            score += 1.0

    return score


# Abbreviation
better = betterEvaluationFunction

