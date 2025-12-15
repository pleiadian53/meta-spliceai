def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

def test(): 
    input_path = '../../requirements.txt'
    install_requires = parse_requirements(input_path)

    print(install_requires)

if __name__ == "__main__": 
    test()
