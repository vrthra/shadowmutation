from typing import List


# def do_char(char: int, key: int) -> int:
#     if char <= 64:
#         return char
#     if char >= 90:
#         return char
#     char = char - 65
#     char = (char + key) % 26
#     char = char + 65
#     return char


def caesar(string: List[int], key: int, decode: bool = False) -> List[int]:
	if decode:
		key = 26 - key
	res = []
	for c in string:
		# c_res = ((c - 65 + key) % 26) + 65
		# c_res = do_char(c, key)
		if c <= 64:
			pass
		elif c >= 90:
			pass
		else:
			c = c - 65
			c = (c + key) % 26
			c = c + 65
		res.append(c)

	return res
 

def test_caesar() -> None:
	msg = "The quick brown fox jumped over the lazy dogs"
	input = [ord(c) for c in msg] # .upper() if ord(c) >= 64 and ord(c) <= 90]
	# print(input)
	enc = caesar(input, 11)
	# print(enc)
	dec = caesar(enc, 11, decode = True)
	# print(dec)
	assert sum(input) == sum(dec)
	
test_caesar()