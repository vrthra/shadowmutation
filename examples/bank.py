class BankAccount:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self._update_overdrawn()
        
    def _update_overdrawn(self):
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
        self._update_overdrawn()

    def withdraw(self, amount):
        self.balance -= amount
        self._update_overdrawn()

    def overdrawn(self):
        return self.overdrawn

def test_accounts():
    my_account = BankAccount(10)

    assert my_account.balance == 10
    assert my_account.overdrawn == False

    my_account.deposit(5)

    assert my_account.balance == 15
    assert my_account.overdrawn == False

    my_account.withdraw(200)
    # try one of each
    #my_account.interest(1)
    #my_account.interest(2)
    assert my_account.balance == -185
    assert my_account.overdrawn == True

test_accounts()
