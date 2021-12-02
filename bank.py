import tainted
original = """
class BankAccount:
    def __init__(self, initial_balance):
        self.balance = initial_balance
    def deposit(self, amount):
        self.balance += amount
    def withdraw(self, amount):
        self.balance -= amount
    def overdrawn(self):
        return self.balance < 0
"""

class BankAccount_:
    def __init__(self, initial_balance):
        # the value indicated by '0' is the main line.
        self.balance = tainted.tint({
            '0':  initial_balance, # mainline, no mutation
            '1.1': initial_balance + 1 # mutation +1
            })

    def deposit(self, amount):
        self.balance = tainted.tint({
            '0':self.balance + amount, # mainline -- no mutation
            '2.1':tainted.untaint(self.balance - amount), # mutation op +/-
            }) + tainted.tint({
            '0':0, # main line -- no mutation
            '2.2':1, # mutation +1
            })

    def withdraw(self, amount):
        self.balance = tainted.tint({
            '0':self.balance - amount, # mainline -- no mutation
            '3.1':tainted.untaint(self.balance + amount), # mutation op +/-
            }) + tainted.tint({
            '0':0, # main line -- no mutation
            '3.2':1, # mutation +1
            }) 


    def interest(self, i):
        self.balance = tainted.tint({
            '0':(self.balance * i), # mainline -- no mutation
            '4.1':tainted.untaint(self.balance / i), # mutation op *//
            })

    def overdrawn(self):
        return self.balance < 0


def test_accounts():
    my_account = BankAccount_(10)
    my_account.deposit(5)
    my_account.withdraw(10)
    # try one of each
    #my_account.interest(1)
    #my_account.interest(2)
    tainted.tassert(my_account.balance == 5)

test_accounts()
