from typing import List


def caesar(string: List[int], key: int, decode: bool = False) -> List[int]:
	if decode:
		key = 26 - key
	res = []
	for c in string:
		if c < 64:
			pass
		elif c > 90:
			pass
		else:
			c = c - 65
			c = c + key
			c = c % 26
			c = c + 65
		res.append(c)

	return res
 

def test_caesar() -> None:
	msg = "The quick brown fox jumped over the lazy dogs"
	input = [ord(c) for c in msg.upper() if ord(c) >= 64 and ord(c) <= 90]
	enc = caesar(input, 11)
	dec = caesar(enc, 11, decode = True)
	assert input == dec
	
test_caesar()