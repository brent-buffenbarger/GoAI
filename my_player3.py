import numpy as np
from os.path import exists
import os

'''
StateTree is a tree structure that uses StateNodes to represent
the minimax tree of possible states
'''
class StateTree:
    def __init__(self, root):
        self.root = root


'''
StateNode holds information about a game state
'''
class StateNode:
    def __init__(self, board, level):
        self.board = board
        self.level = level
        self.move_made = None
        self.children = []
        self.pieces_killed = None

    def add_child(self, board):
        self.children.append(board)


'''
AlphaBetaPlayer is an agent that uses the Alpha Beta Pruning algorithm along
with the Minimax algorithm to play the game of Mini-Go
'''
class AlphaBetaPlayer():
    def __init__(self, piece_type, previous_board, board, depth, branching_factor, steps):
        self.type = piece_type
        self.previous_board = previous_board
        self.board = board
        self.depth = depth
        self.branching_factor = branching_factor
        self.tree = None
        self.komi = len(board) / 2
        self.steps = steps

    def calculate_score(self, board):
        '''
        Calculate the scores for both players on a board

        :param board: The board we are caluclating a score for
        :return: (x_score, o_score) where x_score is the score for Black and o_score is the score for White
        '''

        x_score = 0
        o_score = 0
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 1:
                    x_score += 1
                elif board[i][j] == 2:
                    o_score += 1
        
        return (x_score, o_score + self.komi)

    def detect_neighbor(self, board, row, col):
        '''
        NOTE This function was taken from the detect_neighbor function in host.py

        Detect all the neighbors of a given stone.

        :param board: the board we are considering
        :param row: row number of the board.
        :param col: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        neighbors = []
        # Detect borders and add neighbor coordinates
        if row > 0: neighbors.append((row - 1, col))
        if row < len(board) - 1: neighbors.append((row + 1, col))
        if col > 0: neighbors.append((row, col - 1))
        if col < len(board) - 1: neighbors.append((row, col + 1))
        return neighbors

    def detect_neighbor_ally(self, board, row, col):
        '''
        NOTE This function was taken from the detect_neighbor_ally function in host.py

        Detect the neighbor allies of a given stone.

        :param board: the board we are considering
        :param row: row number of the board.
        :param col: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''

        neighbors = self.detect_neighbor(board, row, col)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[row][col]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, board, row, col):
        '''
        NOTE This function was taken from the ally_dfs function in host.py

        Using DFS to search for all allies of a given stone.

        :param board: the board we are considering
        :param row: row number of the board.
        :param col: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(row, col)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(board, piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, board, row, col):
        '''
        NOTE This function was taken from the find_liberty function in host.py

        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param board: the board we are considering
        :param row: row number of the board.
        :param col: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        ally_members = self.ally_dfs(board, row, col)
        for member in ally_members:
            neighbors = self.detect_neighbor(board, member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False
                    
    def find_dead_pieces(self, board, piece_type):
        '''
        NOTE This function was taken from the find_dead_pieces function in host.py

        Find the died stones that has no liberty in the board for a given piece type.

        :param board: the board we are removing dead pieces from
        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''

        dead_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(board, i, j):
                        dead_pieces.append((i,j))

        return dead_pieces

    def remove_certain_pieces(self, board, positions):
        '''
        NOTE This function was taken from the remove_certain_pieces function in host.py

        Remove the stones of certain locations.

        :param board: the board we are removing dead pieces from
        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        for piece in positions:
            board[piece[0]][piece[1]] = 0

    def remove_dead_pieces(self, board, piece_type, node=None):
        '''
        NOTE This function was taken from the remove_dead_pieces function in host.py

        Remove the dead stones in the board.

        :param board: the board we are removing dead pieces from
        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        dead_pieces = self.find_dead_pieces(board, piece_type)
        if not dead_pieces: return []
        self.remove_certain_pieces(board, dead_pieces)
        if node:
            node.pieces_killed = len(dead_pieces)
        return dead_pieces

    def compare_board(self, board1, board2):
        '''
        NOTE This function was taken from the compare_board function in host.py
        '''

        for i in range(len(board1)):
            for j in range(len(board1)):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def valid_place_check(self, board, row, col, piece_type, node=None):
        '''
        NOTE This function was adapted from the valid_place_check function in host.py

        Check whether a placement is valid.

        :param board: the board
        :param row: row number of the board.
        :param col: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :return: boolean indicating whether the placement is valid.
        '''   

        # Check if the place is in the board range
        if not (row >= 0 and row < len(board)):
            return False
        if not (col >= 0 and col < len(board)):
            return False
        
        # Check if the place already has a piece
        if board[row][col] != 0:
            return False

        # Check if the place has liberty
        board[row][col] = piece_type
        if self.find_liberty(board, row, col):
            return True

        # If not, remove the died pieces of opponent and check again
        dead_pieces = self.remove_dead_pieces(board, 3 - piece_type, node)
        if not self.find_liberty(board, row, col):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if dead_pieces and self.compare_board(self.previous_board, board):
                return False
        return True

    def get_pieces(self, board, piece_type):
        '''
        This function gets all of the active pieces on the board of a certain type
        '''
        pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == piece_type:
                    pieces.append((i, j))
        return pieces

    def largest_group(self, board, piece_type):
        '''
        This function will find the largest group of allies for a given
        stone type and return the number of stones in that group
        '''
        pieces = self.get_pieces(board, piece_type)
        
        largest_group = 0
        for piece in pieces:
            stack = [piece]
            allies = []
            while len(stack) > 0:
                stone = stack.pop()
                allies.append(stone)
                neighbors = self.detect_neighbor_ally(board, stone[0], stone[1])
                for neighbor in neighbors:
                    if neighbor not in stack and neighbor not in allies:
                        stack.append(neighbor)
                largest_group = len(neighbors) if len(neighbors) > largest_group else largest_group
                neighbors = [] if len(neighbors) > largest_group else neighbors

        return largest_group

    def find_liberties(self, board, piece_type):
        '''
        This function finds the total number of liberties a certain
        stone type has
        '''
        pieces = self.get_pieces(board, piece_type)

        liberties = 0
        for piece in pieces:
            for neighbor in self.detect_neighbor(board, piece[0], piece[1]):
                if board[neighbor[0]][neighbor[1]] == 0:
                    liberties += 1
        
        return liberties

    def heuristic(self, state, piece_type):
        '''
        Calculate the heuristic of the given state
        for the given stone type
        '''
        # Current score
        score_weight = 8
        score = self.calculate_score(state.board)
        score_val = score[piece_type - 1] * score_weight
        opp_score_val = score[piece_type - 2] * -score_weight

        # Maximize our own connectedness
        connected_weight = 3
        if self.steps > 14:
            connected_weight = 0
        connected_pieces = self.largest_group(state.board, piece_type)
        connected_val = connected_pieces * connected_weight

        # Minimize the opponents connectedness
        opp_connected_pieces = self.largest_group(state.board, 3 - piece_type)
        opp_connected_val = opp_connected_pieces * -connected_weight

        # Maximize our own liberties
        liberties_weight = 2
        liberties = self.find_liberties(state.board, piece_type)
        liberties_val = liberties_weight * liberties

        # Minimize the opponents liberties
        if self.steps > 14:
            liberties_weight = 4
        opp_liberties = self.find_liberties(state.board, 3 - piece_type)
        opp_liberties_val = opp_liberties * -liberties_weight

        total_score = connected_val + opp_connected_val + liberties_val + opp_liberties_val + score_val + opp_score_val

        return total_score

    def max_value(self, root, alpha, beta, piece_type):
        '''
        This is the part of the minimax algorithm that
        has our agent choosing the state with the highest
        possible score.

        Also uses alpha-beta pruning to build out the tree
        '''
        if root.level == self.depth:
            return self.heuristic(root, 3 - piece_type)

        max_score = float('-inf')
        for i in range(len(root.board)):
            for j in range(len(root.board)):
                new_node = StateNode(root.board.copy(), root.level + 1)
                if self.valid_place_check(new_node.board, i, j, piece_type, new_node):
                    if not new_node.pieces_killed:
                        self.remove_dead_pieces(new_node.board, 3 - piece_type, new_node)
                    new_node.move_made = (i, j)
                    if len(root.children) < self.branching_factor:
                        root.add_child(new_node)
                        max_score = max(max_score, self.min_value(new_node, alpha, beta, 3 - piece_type))
                        if max_score >= beta:
                            return max_score
                        alpha = max(alpha, max_score)

        if len(root.children) == 0:
            return self.heuristic(root, 3 - piece_type)

        return max_score
    
    def min_value(self, root, alpha, beta, piece_type):
        '''
        This is the part of the minimax algorithm that
        has our agent choosing the state with the smallest
        possible score.

        Also uses alpha-beta pruning to build out the tree
        '''
        if root.level == self.depth:
            return self.heuristic(root, 3 - piece_type)

        min_score = float('inf')
        for i in range(len(root.board)):
            for j in range(len(root.board)):
                new_node = StateNode(root.board.copy(), root.level + 1)
                if self.valid_place_check(new_node.board, i, j, piece_type, new_node):
                    if not new_node.pieces_killed:
                        self.remove_dead_pieces(new_node.board, 3 - piece_type, new_node)
                    new_node.move_made = (i, j)
                    if len(root.children) < self.branching_factor:
                        root.add_child(new_node)
                        min_score = min(min_score, self.max_value(new_node, alpha, beta, 3 - piece_type))
                        if min_score <= alpha:
                            return min_score
                        beta = min(beta, min_score)

        if len(root.children) == 0:
            return self.heuristic(root, 3 - piece_type)

        return min_score

    def alpha_beta(self, root):
        '''
        Driver function to start building the minimax tree while
        running alpha-beta pruning to avoid checking states that
        are guaranteed to not be the most desired state
        '''
        max_score = float('-inf')
        beta = float('inf')

        piece_type = 3 - self.type if (root.level % 2) == 0 else self.type
        opponent_type = 3 - piece_type
        optimal_state = None
        for i in range(len(root.board)):
            for j in range(len(root.board)):
                new_node = StateNode(root.board.copy(), root.level + 1)
                if self.valid_place_check(new_node.board, i, j, piece_type, new_node):
                    if not new_node.pieces_killed:
                        self.remove_dead_pieces(new_node.board, opponent_type, new_node)
                    new_node.move_made = (i, j)
                    if len(root.children) < self.branching_factor:
                        root.add_child(new_node)
                        opt_score = self.min_value(new_node, max_score, beta, 3 - self.type)
                        if opt_score > max_score:
                            max_score = opt_score
                            optimal_state = new_node

        return optimal_state

    def starting_moves(self):
        '''
        Function that allows the agent to choose
        an advantageous starting position. The minimax
        algorithm will likely not choose one of these
        locations, so hardcoding the positions works
        best
        '''
        if self.tree.root.board[2][2] == 0:
            return (2, 2)
        elif self.tree.root.board[2][1] == 0:
            return (2, 1)
        
        return None

    def get_input(self):
        '''
        Driver function to get the move from
        the agent.
        '''
        # Hardcoded starting moves
        if self.steps == 1:
            move = self.starting_moves()
            if move and self.valid_place_check(self.tree.root.board, move[0], move[1], self.type):
                return move

        # Build the state tree while running alpha_beta
        optimal_state = self.alpha_beta(self.tree.root)

        if optimal_state:
            return optimal_state.move_made
        else:
            return "PASS"

def is_new_game(board):
    '''
    Function to check to see if we have started a new
    game. That way we can reset the steps counter
    so the agent knows which move it is on.
    '''
    empty_pos = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                empty_pos += 1
    return empty_pos > 23

def read_input(n):
    '''
    NOTE This function was adapted from the read_input function in host.py

    Reads the input file and the steps file to get important game data
    that the agent will use to make decisions
    '''
    path = "input.txt"
    piece_type = None
    previous_board = None
    board = None
    with open(path, 'r') as file:
        lines = file.readlines()

        piece_type = int(lines[0])

        previous_board = np.array([[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]])
        board = np.array([[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]])

    path = "steps.txt"
    steps = 0
    if exists(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            if lines:
                steps = int(lines[0]) + 1

        os.remove(path)

    if is_new_game(board):
        steps = 1
    
    with open(path, 'w') as file:
        file.write(str(steps))
    
    return piece_type, previous_board, board, steps

def write_output(move):
    '''
    NOTE This function was taken from the writeOutput function in host.py

    This function is simply used to write the chosen move to the output
    file so the host file can continue the game.
    '''
    output = "PASS" if move == "PASS" else str(move[0]) + ',' + str(move[1])
    path = "output.txt"
    with open(path, 'w') as file:
        file.write(output)

if __name__ == "__main__":
    board_size = 5
    piece_type, previous_board, board, steps = read_input(board_size)

    if steps < 12:
        ab_depth = 4
    else:
        ab_depth = 6

    player = AlphaBetaPlayer(piece_type, previous_board, board, ab_depth, 25, steps)
    root = StateNode(board, 1)
    player.tree = StateTree(root)
    action = player.get_input()
    write_output(action)