from mixer import *

# generate 10 chairs
chair_to_generate = 10

def main():
    for i in range(chair_to_generate):
        print("Generating Chair: ", i + 1)
        generate(i)

if __name__ == "__main__": main()
