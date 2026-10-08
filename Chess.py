class Chess:
    def __init__(self, x=0, y=0,color=""):
        self.x = x
        self.y = y
        self.color = color
        self.pieces = []
    def add_pieces(self, piece):
         self.pieces.append(piece)
    def move(self):
        pass
    def attack(self):
        pass
    def taken(self):
        pass
    def get_space(self,arr):
        return arr[self.x][self.y]
    
class Pawn(Chess):
    def __init__(self, x, y, first_move=True):
        super().__init__(x,y)
        self.first_move = first_move 
        
    def move(self, space_is_empty, choice):
        if space_is_empty == False:
                self.x = Other_piece.x
                self.y = Other_piece.y
        if self.first_move and choice == 1:
            self.y += 2
            print(self.chess_board[self.x][self.y])
            self.first_move = False
        if space_is_empty:
             self.y += 1
             print(self.chess_board[self.x][self.y])
    def attack(self, space_is_empty,Other_piece):
        if space_is_empty == False:
            self.x = Other_piece.x
            self.y = Other_piece.y
class Bishop(Chess):
    def __init__(self, x, y, color, range_index = 8):
            super().__init__(x,y, color)
            self.range_index = range_index
            
    def move(self, space_is_empty,arr):
        space_list = []
        for i in range(self.range_index):
            next_space = i + 1
            if  next_space + self.x  < 8 and  next_space + self.y < 8:
                if space_is_empty[self.x + next_space][self.y + next_space]:
                    space_list.append(arr[self.x + next_space][self.y + next_space])

            if self.x - next_space  >= 0 and  self.y - next_space >= 0:
                if space_is_empty[self.x - next_space][self.y - next_space]:
                    space_list.append(arr[self.x - next_space][self.y - next_space])

            if  next_space + self.x  < 8 and  self.y - next_space >= 0:
                if space_is_empty[self.x + next_space][self.y - next_space]:
                    space_list.append(arr[self.x + next_space][self.y - next_space])

            if  self.x - next_space  >= 0 and  self.y + next_space < 8:
                if space_is_empty[self.x  - next_space][self.y  + next_space]:
                        space_list.append(arr[self.x - next_space][self.y + next_space])
        return space_list

    def attack(self, space_is_empty,Other_piece):
        if space_is_empty == False:
            self.x = Other_piece.x
            self.y = Other_piece.y
class Knight(Chess):
    def __init__(self, x, y, color):
        super().__init__(x,y, color)
        

    def move(self, space_is_empty,arr):
            space_list = []
            for i in range(2):
                if i == 0:
                    next_space =  1
                    next_space2 = 2
                else:
                    next_space = 2
                    next_space2 = 1
    
                if  next_space2 + self.x  < 8 and  next_space + self.y < 8:
                    if space_is_empty[self.x + next_space2][self.y + next_space]:
                        space_list.append(arr[self.x + next_space2][self.y + next_space])
    
                if self.x - next_space2  >= 0 and  self.y - next_space >= 0:
                    if space_is_empty[self.x - next_space2][self.y - next_space]:
                        space_list.append(arr[self.x - next_space2][self.y - next_space])
    
                if  next_space2 + self.x  < 8 and  self.y - next_space >= 0:
                    if space_is_empty[self.x + next_space2][self.y - next_space]:
                        space_list.append(arr[self.x + next_space2][self.y - next_space])
    
                if  self.x - next_space2  >= 0 and  self.y + next_space < 8:
                    if space_is_empty[self.x  - next_space2][self.y  + next_space]:
                            space_list.append(arr[self.x - next_space2][self.y + next_space])
            return space_list

    def attack(self, space_is_empty,Other_piece):
            if space_is_empty == False:
                self.x = Other_piece.x
                self.y = Other_piece.y
   
class Rook(Chess):
    def __init__(self, x, y, color, range_index = 8):
            super().__init__(x,y, color)
            self.range_index = range_index
             
            
    def move(self, space_is_empty,arr):
        space_list = []
        for i in range(self.range_index):
            next_space = i + 1
            if  next_space + self.x  < 8:
                if space_is_empty[self.x + next_space][self.y]:
                    space_list.append(arr[self.x + next_space][self.y])

            if self.x - next_space  >= 0:
                if space_is_empty[self.x - next_space][self.y]:
                    space_list.append(arr[self.x - next_space][self.y])

            if  self.y - next_space >= 0:
                if space_is_empty[self.x][self.y - next_space]:
                    space_list.append(arr[self.x][self.y - next_space])

            if self.y + next_space < 8:
                if space_is_empty[self.x][self.y  + next_space]:
                        space_list.append(arr[self.x][self.y + next_space])
        return space_list

    def attack(self, space_is_empty,Other_piece):
        if space_is_empty == False:
            self.x = Other_piece.x
            self.y = Other_piece.y
class Queen(Chess):
    def __init__(self, x, y, color, range_index = 8):
            super().__init__(x,y, color)
            self.range_index = range_index
             
            
    def move(self, space_is_empty,arr):
      
        return Rook(self.x,self.y,self.color).move(space_is_empty,arr) + Bishop(self.x,self.y,self.color).move(space_is_empty,arr)

    def attack(self, space_is_empty,Other_piece):
        if space_is_empty == False:
            self.x = Other_piece.x
            self.y = Other_piece.y
class King(Chess):
    def __init__(self, x, y, color, range_index = 1):
            super().__init__(x,y, color)
            self.range_index = range_index
             
            
    def move(self, space_is_empty,arr):
      
        return Rook(self.x,self.y,self.color,self.range_index).move(space_is_empty,arr) + Bishop(self.x,self.y,self.color,self.range_index).move(space_is_empty,arr)

    def attack(self, space_is_empty,Other_piece):
        if space_is_empty == False:
            self.x = Other_piece.x
            self.y = Other_piece.y

def main():
    c = Chess(0,0,"white")
    arr,space_empty = create_Chess_board()
    # Newline after each row
    display_2D_arr(arr)
    print(arr[4][4])
    b = Rook(0,0,"white")
    r = Bishop(0,0, "white") 
    q = Queen(0,0,"white") 
    at = b.move(space_empty,arr)
    #at.sort()
    display(at)
    at = r.move(space_empty,arr)
    display(at)
    at = q.move(space_empty,arr)
    display(at)

   
   

            

def display_2D_arr(arr):
    for row in arr[::-1]:
        for element in row:
            print(element, end=" ")
        print()
def display(arr):
    for element in arr:
        print(element, end=" ")
    print()  

def create_Chess_board():
        arr = []
        is_empty_col =[]
        for column in range(8):
            row_arr = []
            is_empty_row = []
            for row in  range(8):
                row_arr.append(f"{chr(row + 97)}{column + 1}")
                is_empty_row.append(True)
            arr.append(row_arr)
            is_empty_col.append(is_empty_row)

        return arr, is_empty_col
        
#main()