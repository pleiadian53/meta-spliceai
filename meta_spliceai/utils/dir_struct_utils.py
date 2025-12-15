
import fnmatch
import os

def print_directory_structure(root_dir, padding='', include_dirs=None, include_files=None):
    """

    Examples 
    --------
    print_directory_structure('/path/to/directory', include_dirs=['test'], include_files=['*.py'])
    """
    # import fnmatch
    # print(f"root_dir: {root_dir}")
    print(padding[:-1] + '+--' + os.path.basename(root_dir) + '/')
    padding = padding + '   '
    files = []
    if os.path.isdir(root_dir):
        files = sorted(os.listdir(root_dir))
    for file in files:
        path = os.path.join(root_dir, file)
        if os.path.isdir(path):
            if include_dirs is None or any(fnmatch.fnmatch(file, pattern) for pattern in include_dirs):
                print_directory_structure(path, padding + '|  ', include_dirs, include_files)
        else:
            if include_files is None or any(fnmatch.fnmatch(file, pattern) for pattern in include_files):
                print(padding + '|-- ' + file)

    return

def print_directory_structure_v0(root_dir, padding=''):
    print(f"root_dir: {root_dir}")
    print(padding[:-1] + '+--' + os.path.basename(root_dir) + '/')
    padding = padding + '   '
    files = []
    if os.path.isdir(root_dir):
        files = sorted(os.listdir(root_dir))
    for file in files:
        path = os.path.join(root_dir, file)
        if os.path.isdir(path):
            print_directory_structure(path, padding + '|  ')
        else:
            print(padding + '|-- ' + file)

    return

def demo_dirtree(): 
    from dirtree import Dirtree # pip install dir-tree
    from meta_spliceai.system.config import find_project_root
    root_directory = str(find_project_root() / "meta_spliceai")
    dt = Dirtree(root_directory)
    print(dt)

def demo_print_dir_struct(): 
    # Set the root directory you want to print
    from meta_spliceai.system.config import find_project_root
    root_directory = str(find_project_root() / "meta_spliceai")
    include_dirs = ['sphere_pipeline', 'system', 'tests']
    include_files = ['*.py', ]

    print_directory_structure(root_directory, include_dirs=include_dirs, include_files=include_files)


def demo(): 
    demo_print_dir_struct()


if __name__ == "__main__":
    demo()