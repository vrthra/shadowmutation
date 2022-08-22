
class BankAccount:
    balance: int
    overdrawn: bool
    def __init__(self, initial_balance: int):
        self.balance = initial_balance
        self.overdrawn = False
        self.update_overdrawn()

    def update_overdrawn(self) -> None:
        if self.balance >= 0:
            self.overdrawn = False
        else:
            self.overdrawn = True

    def deposit(self, amount: int) -> None:
        self.balance = self.balance + amount
        self.update_overdrawn()

    def withdraw(self, amount: int) -> None:
        self.balance = self.balance - amount
        self.update_overdrawn()

    def is_overdrawn(self) -> bool:
        return self.overdrawn


def test_accounts() -> None:
    my_account = BankAccount(10)
    assert my_account.balance == 10
    assert my_account.overdrawn == False
    my_account.deposit(5)
    assert my_account.balance == 15
    assert my_account.overdrawn == False
    my_account.withdraw(200)
    assert my_account.balance == -185
    assert my_account.overdrawn == True


test_accounts()
