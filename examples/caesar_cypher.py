from typing import List

def caesar(string: List[int], key: int, decode: bool = False) -> List[int]:
	if decode:
		key = 26 - key
	res = []
	for c in string:
		# c_res = ((c - 65 + key) % 26) + 65
		c_res = c - 65
		c_res = c_res + key
		c_res = c_res % 26
		c_res = c_res + 65
		res.append(c_res)

	return res
 

def test_caesar() -> None:
	msg = "The quick brown fox jumped over the lazy dogs"
	input = [ord(c) for c in msg.upper() if ord(c) >= 64 and ord(c) <= 90]
	# print(input)
	enc = caesar(input, 11)
	# print(enc)
	dec = caesar(enc, 11, decode = True)
	# print(dec)
	assert sum(input) == sum(dec)
	
test_caesar()