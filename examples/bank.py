from typing import Tuple

# class BankAccount:
#     balance: int
#     overdrawn: bool
#     def __init__(self, initial_balance: int):
#         self.balance = initial_balance
#         self.overdrawn = False
#         self._update_overdrawn()

#     def _update_overdrawn(self) -> None:
#         if self.balance >= 100:
#             print('not overdrawn')
#             self.overdrawn = False
#         elif self.balance >= 0:
#             print('all good')
#             self.overdrawn = False
#         elif self.balance < -100:
#             print('very overdrawn')
#             self.overdrawn = True
#         else:
#             self.overdrawn = True

#     def deposit(self, amount: int) -> None:
#         self.balance += amount
#         self._update_overdrawn()

#     def withdraw(self, amount: int) -> None:
#         self.balance -= amount
#         self._update_overdrawn()

#     def is_overdrawn(self) -> bool:
#         return self.overdrawn


# def test_accounts() -> None:
#     my_account = BankAccount(10)
#     assert my_account.balance == 10
#     assert my_account.overdrawn == False
#     my_account.deposit(5)
#     assert my_account.balance == 15
#     assert my_account.overdrawn == False
#     my_account.withdraw(200)
#     assert my_account.balance == -185
#     assert my_account.overdrawn == True


def is_overdrawn(balance: int) -> bool:
    if balance >= 100:
        print('not overdrawn')
        overdrawn = False
    elif balance >= 0:
        print('all good')
        overdrawn = False
    elif balance < -100:
        print('very overdrawn')
        overdrawn = True
    else:
        overdrawn = True
    return overdrawn

def deposit(balance: int, amount: int) -> Tuple[int, bool]:
    balance += amount
    overdrawn = is_overdrawn(balance)
    return balance, overdrawn

def withdraw(balance: int, amount: int) -> Tuple[int, bool]:
    balance -= amount
    overdrawn = is_overdrawn(balance)
    return balance, overdrawn


def test_accounts() -> None:
    balance = 10
    assert is_overdrawn(balance) == False

    balance, overdrawn = deposit(balance, 5)
    assert balance == 15
    assert overdrawn == False

    balance, overdrawn = withdraw(balance, 200)
    assert balance == -185
    assert overdrawn == True


test_accounts()
