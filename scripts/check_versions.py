"""Quick script to check installed versions of required packages"""
import pkg_resources

PACKAGES = [
    'polars',
    'pandas',
    'pybedtools',
    'gffutils',
    'biopython',
    'tqdm',
    'tensorflow',
    'keras',
    'spliceai',
    'pyfaidx',
    'xgboost',
    'shap',
    'h5py',
    'tensorboard',
    'rich',
    'matplotlib',
    'seaborn'
]

def get_package_version(package_name):
    try:
        if package_name == 'biopython':
            package_name = 'bio'
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not installed"

if __name__ == "__main__":
    print("\nInstalled Package Versions:")
    print("-" * 50)
    for package in PACKAGES:
        version = get_package_version(package)
        print(f"{package:<15} {version}")
