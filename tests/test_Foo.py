import unittest

class FooTest(unittest.TestCase):
	def test(self):
		self.failIf(True)

def main():
	unittest.main()

if __name__ == '__main__':
	main()