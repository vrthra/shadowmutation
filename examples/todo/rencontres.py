

# Returns value of Binomial Coefficient C(n, k)
def binomial_coefficient(n: int, k: int) -> int:
 
    # Base Cases
    if (k == 0 or k == n):
        return 1
 
    # Recurrence relation
    return (binomial_coefficient(n - 1, k - 1)
          + binomial_coefficient(n - 1, k))
 
# Return Recontres number D(n, m)
def rencontres_number(n: int, m: int) -> int:
 
    # base condition
    if (n == 0 and m == 0):
        return 1
 
    # base condition
    if (n == 1 and m == 0):
        return 0
 
    # base condition
    if (m == 0):
        return ((n - 1) * (rencontres_number(n - 1, 0)
                         + rencontres_number(n - 2, 0)))
 
    return (binomial_coefficient(n, m) *
            rencontres_number(n - m, 0))
 

def test_rencontres() -> None:
    n = 7
    m = 2
    res = rencontres_number(n, m)
    assert res == 924

test_rencontres()
