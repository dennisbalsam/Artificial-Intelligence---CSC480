board_size = 8
board = [[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]

# This takes the current board and a col, and returns true if a queen is placed
def place_queen(board, col):
    
    # Bool val. keeps track if a queen has been placed or not
    queen_placed = False

    for row in range(board_size):
        # If there is an empty space and it is not threatened by a queen, in that location
        if(board[row][col] == 0 and validate_placement(board, row, col)):
            board[row][col] = 1
            queen_placed = True
            break
    
    return queen_placed

# This ensures that the placement of a queen is not threatened by another queen
def validate_placement(board, row, col):
    
    #Checking horizontal and vertical
    #--------REVIEW -may be screwy
    for x in range(board_size):
        #row
        if board[row][x] == 1:
            return False
        #col
        if board[x][col] == 1:
            return False
        
    #check diag top left
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    #check diag down right
    i = row + 1
    j = col + 1
    while i <= 7 and j <= 7:
        if board[i][j] == 1:
            return False
        i += 1
        j += 1
    
    #check diag top right
    i = row - 1
    j = col + 1
    while i >= 0 and j <= 7:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1
    
    #check diag down left
    i = row + 1
    j = col - 1
    while i <= 7 and j >= 0:
        if board[i][j] == 1:
            return False
        i += 1
        j -= 1
    
    return True

def print_board(board):
    for x in board:
        print(x)

# This is what happens when we ask to place 8 queens. [CHECK IMAGE] Only 5 are placed.
# Currently, the program only allows for direct placements and will stop if there are 
# no more open spaces. What we want to happen is [CHECK IMAGE2]
#---------------------------------------------
#
# if queen can be placed:
#     place queen in first available row position     
# else (if queen can't be placed):
#     remove current queen
#     move prev. queen further in its row
# 
#---------------------------------------------
# This is done recursively until we can't continue placing queens. Just having
# trouble figuring out the recursion 

#EXAMPLE
place_queen(board, 0)
place_queen(board, 1)
place_queen(board, 2)
place_queen(board, 3)
place_queen(board, 4)
#NO MORE AVAILABLE SPOTS AFTER THIS POINT
place_queen(board, 5)
place_queen(board, 6)
place_queen(board, 7)

print_board(board)