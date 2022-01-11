def discriminant(a: int, b: int, c: int) -> int:
 
    discriminant = (b**2) - (4*a*c)
    if discriminant > 0:
         
        print('Discriminant is', discriminant,
                "which is Positive")
                 
        print('Hence Two Solutions')
         
    elif discriminant == 0:
         
        print('Discriminant is', discriminant,
                "which is Zero")
                 
        print('Hence One Solution')
         
    elif discriminant < 0:
         
        print('Discriminant is', discriminant,
                "which is Negative")
                 
        print('Hence No Real Solutions')
 

def test_discriminant():
    # Driver Code
    a = 20
    b = 30
    c = 10
    discriminant(a, b, c)


test_discriminant()