import tainted
class BankAccount:
    def __init__(self, initial_balance):
        # the value indicated by '0' is the main line.
        self.balance = tainted.tint({
            '0':  initial_balance, # mainline, no mutation
            '1.1': initial_balance + 1 # mutation +1
            })
        if tainted.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True

    def deposit(self, amount):
        self.balance = tainted.tint({
            '0':self.balance + amount, # mainline -- no mutation
            '2.1':tainted.untaint(self.balance - amount), # mutation op +/-
            }) + tainted.tint({
            '0':0, # main line -- no mutation
            '2.2':1, # mutation +1
            })
        if tainted.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True


    def withdraw(self, amount):
        self.balance = tainted.tint({
            '0':self.balance - amount, # mainline -- no mutation
            '3.1':tainted.untaint(self.balance + amount), # mutation op +/-
            }) + tainted.tint({
            '0':0, # main line -- no mutation
            '3.2':1, # mutation +1
            })
        if tainted.cond(self.balance >= 0):
            self.overdrawn = False
        else:
            self.overdrawn = True

    def interest(self, i):
        self.balance = tainted.tint({
            '0':(self.balance * i), # mainline -- no mutation
            '4.1':tainted.untaint(self.balance / i), # mutation op *//
            })

    def overdrawn(self):
        return self.overdrawn


def test_accounts():
    my_account = BankAccount(0)
    my_account.deposit(5)
    my_account.withdraw(5)
    # try one of each
    #my_account.interest(1)
    #my_account.interest(2)
    tainted.tassert(my_account.balance == 0)

test_accounts()
