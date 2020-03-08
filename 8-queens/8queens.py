''' 8 queens problem is a famous problem involving the placing of 8 queens on a 8x8 chessboard
such that none of the 8 queens are able to atack each other. There are a total of 92 solutions to this problem '''

# use time library to track how long finding all solutions takes
import time

# board size
board_size = 8
totalSolutions = 0

# This takes the current board and a col, and returns true if a queen is able to be placed
def place_queen(board, col):

    global timeStart
    global totalSolutions
    # Bool val. keeps track if a queen has been placed or not
    queens_placed = False

    # column is 8 so this is a valid solution
    if col == 8:
        # increase solutions found
        totalSolutions +=1
        # check for time used to print and remove it
        timeForPrint = time.time()
        print('-----------', totalSolutions, '-----------')
        print_board(board)
        timeStart = timeStart - (timeForPrint - time.time())

        return True 

    # go through every row in the column untila safe place is found
    for row in range(board_size):
        # If there is an empty space and it is not threatened by a queen, in that location
        if(board[row][col] == 0 and validate_placement(board, row, col)):
            # if validate_placement returns true we place a queen in that row
            board[row][col] = 1
            # use recursion to keep traversing the chess board and placing queens, this will lead to next column, if the recursion continues until the column is 8, then we have found a solution
            queens_placed = place_queen(board,col +1) or queens_placed
            # if the recusrion fails to place all 8 queens, we will escape the previous lines calls, and reset the current marked spot to empty and try the next rows until valid solution is found
            board[row][col] = 0  # essentially this allows for backtracking

    
    # will return false if no valid solution is found, otherwise it will return true and a solution is found
    return queens_placed

# This ensures that the placement of a queen is not threatened by another queen
def validate_placement(board, row, col):
    #Checking horizontal row 
    for x in range(col):
        #row
        if board[row][x] == 1:
            return False
        #col -------------- column also doesnt have to be checked, we use 1 column per queen anyways
        # if board[x][col] == 1:
        #     return False
        
    #check diag top left
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1


    
    #check diag down left
    i = row + 1
    j = col - 1
    while i <= 7 and j >= 0:
        if board[i][j] == 1:
            return False
        i += 1
        j -= 1
    
    ''' We dont have to check the right side since were placing starting form the left, so attacking queens will only be there 
    # #check diag down right
    # i = row + 1
    # j = col + 1
    # while i <= 7 and j <= 7:
    #     if board[i][j] == 1:
    #         return False
    #     i += 1
    #     j += 1
    
    # #check diag top right
    # i = row - 1
    # j = col + 1
    # while i >= 0 and j <= 7:
    #     if board[i][j] == 1:
    #         return False
    #     i -= 1
    #     j += 1
    '''
    return True

# function to print out the board
def print_board(board):
    for x in board:
        print(x)
    print()

# -------------------------------------- driver ----------------------------------------------
# create the board using simple for loop
board= []
for j in range(8):
    row = []
    for i in range(8):
        row.append(0)
    board.append(row)

#mark the starting time
timeStart = time.time()
totalTime = 0
# find the solutions
foundSolutions = place_queen(board,0)

# mark ending time before printing
totalTime = time.time() - timeStart

if foundSolutions == False:
    ("print there are no solutions")

print("The total time used to find all solutions is: ", totalTime)


