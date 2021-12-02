import shadow
class BankAccount:
    def __init__(self, initial_balance):
        # the value indicated by '0' is the main line.
        self.balance = shadow.tint({
            '0':  initial_balance, # mainline, no mutation
            '1.1': initial_balance + 1 # mutation +1
            })
        if shadow.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True

    def deposit(self, amount):
        self.balance = shadow.tint({
            '0':self.balance + amount, # mainline -- no mutation
            '2.1':shadow.untaint(self.balance - amount), # mutation op +/-
            }) + shadow.tint({
            '0':0, # main line -- no mutation
            '2.2':1, # mutation +1
            })
        if shadow.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True


    def withdraw(self, amount):
        self.balance = shadow.tint({
            '0':self.balance - amount, # mainline -- no mutation
            '3.1':shadow.untaint(self.balance + amount), # mutation op +/-
            }) + shadow.tint({
            '0':0, # main line -- no mutation
            '3.2':1, # mutation +1
            })
        if shadow.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True

    def interest(self, i):
        self.balance = shadow.tint({
            '0':(self.balance * i), # mainline -- no mutation
            '4.1':shadow.untaint(self.balance / i), # mutation op *//
            })

def test_accounts():
    my_account = BankAccount(0)
    my_account.deposit(5)
    my_account.withdraw(5)
    # try one of each
    #my_account.interest(1)
    #my_account.interest(2)
    shadow.tassert(my_account.balance == 0)
    shadow.tassert(my_account.overdrawn == False)

test_accounts()
