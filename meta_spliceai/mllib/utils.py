import os
import pandas as pd
import matplotlib.pyplot as plt


# hpbandster_wrapper.py
def import_hpbandster_search_cv():
    try:
        from hpbandster_sklearn import HpBandSterSearchCV
        return HpBandSterSearchCV
    except ImportError as e:
        error_message = str(e)
        if "_check_fit_params" in error_message:
            print(f"ImportError: {e}. Applying monkey patch for _check_fit_params.")
            
            # Import the necessary function from compat.py
            from .compat import _check_method_params
            
            # Monkey patch the sklearn.utils.validation module
            print("[action] Patching sklearn.utils.validation._check_fit_params")
            import sklearn.utils.validation
            sklearn.utils.validation._check_fit_params = _check_method_params
            
            # Try importing HpBandSterSearchCV again after the patch
            from hpbandster_sklearn import HpBandSterSearchCV
            return HpBandSterSearchCV
        else:
            raise e

def print_emphasized(text, style='bold', edge_effect=True, symbol='='):
    styles = {
        'bold': '\033[1m',
        'underline': '\033[4m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    end_style = '\033[0m'  # Reset to default style

    if edge_effect:
        edge_line = symbol * len(text)
        print(edge_line)
        print(f"{styles.get(style, '')}{text}{end_style}")
        print(edge_line)
    else:
        print(f"{styles.get(style, '')}{text}{end_style}")


def highlight(message=None, symbol='=', prefix=None, n=80, adaptive=False, border=0, offset=5, stdout=True): 
    output = ''
    if border is not None: output = '\n' * border
    # output += symbol * n + '\n'
    if isinstance(message, str) and len(message) > 0:
        if prefix: 
            line = '%s: %s\n' % (prefix, message)
        else: 
            line = '%s\n' % message
        if adaptive: n = len(line)+offset 

        output += symbol * n + '\n' + line + symbol * n
    elif message is not None: 
        # message is an unknown object of some class
        if prefix: 
            line = '%s: %s\n' % (prefix, str(message))
        else: 
            line = '%s\n' % str(message)
        if adaptive: n = len(line)+offset 
        output += symbol * n + '\n' + line + symbol * n
    else: 
        output += symbol * n
        
    if border is not None: 
        output += '\n' * border
    if stdout: 
        print(output)
    return output


def dataframes_equal(df1, df2):
    # Basic checks

    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame), \
        f"type(df1):{type(df1)} <> type(df2): {type(df2)}"

    if not df1.columns.equals(df2.columns):
        return False
    if df1.shape != df2.shape:
        return False

    for col in df1.columns:
        col1 = df1[col]
        col2 = df2[col]

        for idx in df1.index:
            val1 = col1.at[idx]
            val2 = col2.at[idx]

            if isinstance(val1, pd.DataFrame) or isinstance(val2, pd.DataFrame):
                if not dataframes_equal(val1, val2):  # recursive check
                    return False
            else:
                if pd.isna(val1) and pd.isna(val2):
                    continue
                if val1 != val2:
                    return False

    return True


def savefig(plt, fpath, ext='tif', dpi=500, message='', verbose=True):
    """
    fpath: 
       name of output file
       full path to the output file

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os

    # [todo] Configuration
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    output_dir = ''
    fname = 'generic-figure.%s' % ext
    if fpath and os.path.isdir(fpath): 
        # file name is not given, just an output dir
        outputdir = fpath
    else: 
        # not a directory, assuming it's a full path for which the file may not exist yet
        outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

        if fname: 
            # automatically infer ext
            if not ext: 
                # But this logic can be problematic because the file name itself may contain '.'
                fext = fname.split('.')
                if len(fext) >= 2: 
                    ext = fext[-1] # use the given file extension as the preferred extension
            else: 
                # Check if the input file name already includes the extension
                n = len(ext)
                if fname[-n:] == ext and fname[-(n+1)] == '.': 
                    # no-opt
                    pass
                else: 
                    fname = f"{fname}.{ext}"

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir

    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    assert ext in supported_formats, "Unsupported graphic format: %s" % ext

    # fbase, fext = os.path.splitext(fname)
    # if len(fext) == 0: # file name does not contain extension
    #     fname = f"{fname}.{ext}"
    
    fpath = os.path.join(outputdir, fname)

    if verbose: print('(save_figure) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # NOTE: pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return