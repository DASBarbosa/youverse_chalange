import sys

def read_stdin() -> str:
    print (len(sys.argv))
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to your image")
        sys.exit(1)
    elif len(sys.argv) > 2:
        raise ValueError("Only one argument accepted, image path")
    return sys.argv[1]