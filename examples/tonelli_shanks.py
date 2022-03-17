def pow(base: int, exp: int, mod: int) -> int:
    res: int = base ** exp
    res = res % mod
    return res

def legendre_symbol(a: int, p: int) -> int:
    """
    Legendre symbol
    Define if a is a quadratic residue modulo odd prime
    http://en.wikipedia.org/wiki/Legendre_symbol
    """
    ls = p - 1
    ls = ls//2
    ls = pow(a, ls, p)
    p_less = p - 1
    if ls == p_less:
        return -1
    return ls

def prime_mod_sqrt(a: int, p: int) -> list[int]:
    """
    Square root modulo prime number
    Solve the equation
        x^2 = a mod p
    and return list of x solution
    http://en.wikipedia.org/wiki/Tonelli-Shanks_algorithm
    """
    a = a % p

    # Simple case
    if a == 0:
        return [0]
    if p == 2:
        return [a]

    # Check solution existence on odd prime
    leg_sym = legendre_symbol(a, p)
    if leg_sym != 1:
        return []

    # Simple case
    p_mod = p % 4
    if p_mod == 3:
        x = p + 1
        x = x//4
        x = pow(a, x, p)
        return [x, p-x]

    # Factor p-1 on the form q * 2^s (with Q odd)
    q = p - 1
    s = 0
    max_iter = 10
    while True:
        if max_iter <= 0:
            break
        q_mod = q % 2
        if q_mod != 0:
            break
        s = s + 1
        q = q // 2
        max_iter = max_iter - 1

    # Select a z which is a quadratic non resudue modulo p
    z = 1
    max_iter = 10
    while True:
        if max_iter <= 0:
            break
        leg_sym = legendre_symbol(z, p)
        if leg_sym == -1:
            break
        z = z + 1
        max_iter = max_iter - 1
    c = pow(z, q, p)

    # Search for a solution
    x = q + 1
    x = x//2
    x = pow(a, x, p)
    t = pow(a, q, p)
    m = s
    max_iter_outer = 10
    while True:
        if max_iter_outer <= 0:
            break
        if t == 1:
            break
        # Find the lowest i such that t^(2^i) = 1
        i = 0
        e = 2
        i = 1
        max_iter_inner = 10
        while True:
            if max_iter_inner <= 0:
                break
            if i > m:
                break
            pp = pow(t, e, p)
            if pp == 1:
                break
            e = e * 2
            i = i + 1
            max_iter_inner = max_iter_inner - 1

        # Update next value to iterate
        b = m - i
        b = b - 1
        b = 2**b
        b = pow(c, b, p)

        x = x * b
        x = x % p

        t = t * b
        t = t * b
        t = t % p

        c = b * b
        c = c % p

        m = i
        max_iter_outer = max_iter_outer - 1

    return [x, p-x]


def test_tonelli_shanks() -> None:
    res = prime_mod_sqrt(5, 41)
    assert res == [28, 13]


test_tonelli_shanks()