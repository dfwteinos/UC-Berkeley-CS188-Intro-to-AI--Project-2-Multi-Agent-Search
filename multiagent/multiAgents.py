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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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

        Food=newFood.asList()
        Ghosts= successorGameState.getGhostPositions()
        MinFood=float("inf")
        GhostA=float("inf")

        #If pacman ate all his food
        if (len(Food)==0):
          return float("inf")

        #If our agent is too close to ghosts

        for i in Ghosts:

          if ( manhattanDistance(newPos,i) <2 ):
            return -float("inf")
          
          elif ( ((manhattanDistance(newPos,i))>=2) and ((manhattanDistance(newPos,i))<4) ):
            GhostA=0.5
          else:
            GhostA=0.1
      
        #Eating food when ghosts are not near
        
        for j in Food:
          MinFood=min(MinFood,manhattanDistance(newPos,j))

        #Our Initial return State + some food evaluation function + Ghost phobia

        return successorGameState.getScore() + 1.0/MinFood - 1.0/GhostA
        
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        
        def Minimax(gameState,Depth,agentIndex):

          if agentIndex>=gameState.getNumAgents():
            Depth+=1
            agentIndex=0

          if (Depth==self.depth):
            return self.evaluationFunction(gameState)

          ###############################################F# => MAX VALUE

          if (agentIndex==0):                                   #If we are on Pacman
            LegalActions=gameState.getLegalActions(agentIndex)
            MaxEvaluation=["MAX",-float("inf")]

            if not LegalActions:
              return self.evaluationFunction(gameState)         #If we don't have any other move,return the evaluation function

            for Actions in LegalActions:
              CurrentEvaluation=Minimax( gameState.generateSuccessor(agentIndex,Actions) ,Depth,agentIndex+1)   #AgentIndex+1 means we go down and we traverse the tree
            
              if type(CurrentEvaluation) is not list:           #We must be right at the type
                if  CurrentEvaluation>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation]

              else:
                if  CurrentEvaluation[1]>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation[1]]
            
            return MaxEvaluation
          
          ################################################# => MIN VALUE

          else:
            LegalActions=gameState.getLegalActions(agentIndex)
            MinEvaluation=["MIN",float("inf")]

            if not LegalActions:
              return self.evaluationFunction(gameState)

            for Actions in LegalActions:
              CurrentEvaluation=Minimax( gameState.generateSuccessor(agentIndex,Actions) ,Depth,agentIndex+1)

              if type(CurrentEvaluation) is not list:
                if  CurrentEvaluation<MinEvaluation[1]:
                  MinEvaluation=[Actions,CurrentEvaluation]
            
              else:
                if  CurrentEvaluation[1]<MinEvaluation[1]:
                  MinEvaluation=[Actions,CurrentEvaluation[1]]

          return MinEvaluation
         
        Actions= Minimax(gameState,0,0)               #Return the right Actions
        return Actions[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def Minimax(gameState,Depth,alpha,beta,agentIndex):
          
          if agentIndex>=gameState.getNumAgents():
            Depth+=1
            agentIndex=0

          if (Depth==self.depth):
            return self.evaluationFunction(gameState)
          
          if (agentIndex==0):

            ################################################################=>Max Value            
            LegalActions=gameState.getLegalActions(agentIndex)
            MaxEvaluation=["MAX",-float("inf")]

            if not LegalActions:                            #If we don't have any other move,return the evaluation function
              return self.evaluationFunction(gameState)

            for Actions in LegalActions:
              CurrentEvaluation=Minimax( gameState.generateSuccessor(agentIndex,Actions) ,Depth,alpha,beta,agentIndex+1)
            
              if type(CurrentEvaluation) is not list:
                if  CurrentEvaluation>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation]
                if (CurrentEvaluation>beta):
                
                  MaxEvaluation=[Actions,CurrentEvaluation]
                  return MaxEvaluation
              
                alpha=max(alpha,CurrentEvaluation)

              else:
                if  CurrentEvaluation[1]>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation[1] ]
                if (CurrentEvaluation[1]>beta):
                
                  MaxEvaluation=[Actions,CurrentEvaluation[1]]
                  return MaxEvaluation
                alpha=max(alpha,CurrentEvaluation[1])
          
            return MaxEvaluation
          
          
          else :
          
            ############################################################# => MIN_VALUE
            LegalActions=gameState.getLegalActions(agentIndex)
            MinEvaluation=["MIN",float("inf")]

            if not LegalActions:
              return self.evaluationFunction(gameState)         #If we don't have any other move,return the evaluation function

            for Actions in LegalActions:
              CurrentEvaluation=Minimax( gameState.generateSuccessor(agentIndex,Actions) ,Depth,alpha,beta,agentIndex+1)

              if type(CurrentEvaluation) is not list:
                if  CurrentEvaluation<MinEvaluation[1]:
                  MinEvaluation=[Actions,CurrentEvaluation]
                if CurrentEvaluation<alpha:
                
                  MinEvaluation=[Actions,CurrentEvaluation]
                  return MinEvaluation                          #Here we chop
            
                beta=min(beta,CurrentEvaluation)
            
              else:
                if  CurrentEvaluation[1]<MinEvaluation[1]:
                  MinEvaluation=[Actions,CurrentEvaluation[1]]
                if CurrentEvaluation[1]<alpha:                  #Here we chop

                  MinEvaluation=[Actions,CurrentEvaluation[1]]
                  return MinEvaluation
              
                beta=min(beta,CurrentEvaluation[1])
               
          return MinEvaluation
          
        Alpha=-float("inf")
        Beta=float("inf")
        Actions=Minimax(gameState,0,Alpha,Beta,0)
        return Actions[0]

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

        def Expectimax(gameState,Depth,agentIndex):
          
          if agentIndex>=gameState.getNumAgents():
            Depth+=1
            agentIndex=0

          if (Depth==self.depth):                                 #If we are on Leaves
            return self.evaluationFunction(gameState)


          ############################################################# => Max Value,same as Q2

          if (agentIndex==0):
            LegalActions=gameState.getLegalActions(agentIndex)
            MaxEvaluation=["MAX",-float("inf")]

            if not LegalActions:
              return self.evaluationFunction(gameState)

            for Actions in LegalActions:
              CurrentEvaluation=Expectimax( gameState.generateSuccessor(agentIndex,Actions) ,Depth,agentIndex+1)
            
              if type(CurrentEvaluation) is not list:
                if  CurrentEvaluation>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation]

              else:
                if  CurrentEvaluation[1]>MaxEvaluation[1]:
                  MaxEvaluation=[Actions,CurrentEvaluation[1]]
            
            return MaxEvaluation
          
          else:          
            
            ############################################################# => Except Value,returns the Probability of the Node

            LegalActions=gameState.getLegalActions(agentIndex)
            ExpectVal=["EXP",0]
            Cases=len(LegalActions)                               #The number of the childs
            if (Cases):
              Probability=1.0/Cases                               #All the nodes have the same probability to show up

            if not LegalActions:
              return self.evaluationFunction(gameState)
          
            for Actions in LegalActions:
              CurrentEvaluation=Expectimax( gameState.generateSuccessor(agentIndex,Actions),Depth,agentIndex+1)

              if type(CurrentEvaluation) is list:
            
                ExpectVal[0]=Actions
                ExpectVal[1]+=CurrentEvaluation[1]*Probability
          
              else:
              
                ExpectVal[0]=Actions
                ExpectVal[1]+=CurrentEvaluation*Probability

            return ExpectVal
            
        Actions= Expectimax(gameState,0,0)
        return Actions[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Please,check README for this function!
      
    """
    "*** YOUR CODE HERE ***"

    Position=currentGameState.getPacmanPosition()       
    Food=currentGameState.getFood().asList()
    FoodCheck=-float("inf")                                     #Initialize
    GhostCheck=float("inf")

    for food in Food:                                       
      MinFood= manhattanDistance(Position,food)*(-1)            #If we are at the best case: The food is next to us! This will return Eval-1
      if MinFood>FoodCheck:
        FoodCheck=MinFood

    if FoodCheck==-float("inf"):
      FoodCheck=0

    return currentGameState.getScore() + FoodCheck 

# Abbreviation
better = betterEvaluationFunction

