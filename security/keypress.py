import keyboard

def pfun():
    print("you pressed p")
def rfun():
    print("you pressed r")    

while True:
    if keyboard.read_key()=="p":
        #pfun()
        print(" you pressed p")
    if keyboard.read_key()=="r":   
        print(" you pressed r")
        #rfun()   
    if keyboard.read_key()=="q":
        print("You pressed q")
        break