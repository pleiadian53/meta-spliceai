import os, sys

# Import dynamic project root detection
from meta_spliceai.system.config import Config


class ExperimentTracker(object):

    home_dir = os.path.expanduser("~")
    # Use dynamic project root instead of hardcoded path
    data_prefix = Config.PROJ_DIR
    experiment_root = 'experiments'
    # Options: "/mnt/SpliceMediator/splice-mediator/experiments/"
    #          f"{home_dir}/work/nmd/experiments" 

    def __init__(self, experiment="principal_isoforms", 
                    model_type="descriptor", model_name=None):
        self.experiment = experiment
        self.model_type = model_type  # e.g. descriptor, sequence 
        self.model_name = model_name  # e.g. xgboost, transformer
        self.check_datadir()

    def check_datadir(self): 

        # Standardize the root of data prefix
        basename = os.path.basename(ExperimentTracker.data_prefix)
        experiment_root = ExperimentTracker.experiment_root
        if basename != ExperimentTracker.experiment_root: 
            ExperimentTracker.data_prefix = f"{ExperimentTracker.data_prefix}/{experiment_root}"

        # Other rules ...

    @property
    def experiment_dir(self): 
        parent_dir = os.path.join(ExperimentTracker.data_prefix, self.experiment) 
        # Path(parent_dir).mkdir(parents=True, exist_ok=True)
        expr_dir = os.path.join(parent_dir, self.model_type)

        if self.model_name is not None: 
            expr_dir = os.path.join(expr_dir, self.model_name)
        return expr_dir