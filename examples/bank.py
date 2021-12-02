class BankAccount:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        if self.balance >= 100:
            print("not overdrawn")
            self.overdrawn = False
        elif self.balance >= 0:
            print("all good")
            self.overdrawn = False
        elif self.balance < -100:
            print("very overdrawn")
            self.overdrawn = True
        else:
            self.overdrawn = True

    def deposit(self, amount):
        self.balance += amount
        if self.balance >= 0:
            self.overdrawn = False
        else:
            self.overdrawn = True

    def withdraw(self, amount):
        self.balance -= amount
        if self.balance >= 0:
            self.overdrawn = False
        else:
            self.overdrawn = True

    def overdrawn(self):
        return self.overdrawn

def test_accounts():
    my_account = BankAccount(10)
    my_account.deposit(5)
    my_account.withdraw(10)
    # try one of each
    #my_account.interest(1)
    #my_account.interest(2)
    assert my_account.balance == 5

test_accounts()
