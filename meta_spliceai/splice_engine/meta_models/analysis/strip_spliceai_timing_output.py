import re

def strip_debug_messages(log_text):
    """
    Removes debug message lines from log text that are prefixed with common
    debug indicators like [DEBUG], [debug], (debug), etc.

    Parameters
    ----------
    log_text : str
        The original log text containing debug messages.

    Returns
    -------
    str
        Cleaned log text with debug messages removed.
    """
    # Match common debug message formats
    debug_patterns = [
        # Match [DEBUG] or [debug] format
        r'^\s*\[\s*(?:DEBUG|debug|Debug)\s*\].*$',
        # Match (DEBUG) or (debug) format
        r'^\s*\(\s*(?:DEBUG|debug|Debug)\s*\).*$',
        # Match DEBUG: or debug: format
        r'^\s*(?:DEBUG|debug|Debug)\s*:.*$',
        # Match #DEBUG# or #debug# format
        r'^\s*#\s*(?:DEBUG|debug|Debug)\s*#.*$',
        # Match DEBUG - or debug - format
        r'^\s*(?:DEBUG|debug|Debug)\s*-.*$',
        # Match lines with "debug" as first word on the line
        r'^\s*(?:DEBUG|debug|Debug)\b.*$',
        # Match lines with color codes for debug messages
        r'^.*\[\d+m\s*(?:DEBUG|debug|Debug).*$',
    ]
    
    # Combine all patterns with OR (|)
    combined_pattern = re.compile('|'.join(f'({pattern})' for pattern in debug_patterns), re.MULTILINE)
    cleaned_text = combined_pattern.sub('', log_text)
    
    # Remove potential multiple consecutive empty lines left behind
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text).strip()
    
    return cleaned_text


def strip_spliceai_timing_logs(log_text):
    """
    Removes SpliceAI timing output lines from log text, handling both
    plain text and ANSI escape sequences that appear in terminal logs.

    Parameters
    ----------
    log_text : str
        The original log text containing redundant SpliceAI timing information.

    Returns
    -------
    str
        Cleaned log text with timing lines removed.
    """
    # First handle the simple visible text pattern (what appears on screen)
    timing_pattern = re.compile(r'^\d+/\d+\s+━+\s+\d+s\s+\d+ms/step$', re.MULTILINE)
    cleaned_text = timing_pattern.sub('', log_text)
    
    # Now handle the complex ANSI escape sequence pattern
    # This matches lines with control characters like ^M (carriage return) and ANSI color codes
    ansi_timing_pattern = re.compile(
        r'^\^M\^\[\[\d+m\d+/\d+\^\[\[0m\s+\^\[\[32m━+\^\[\[0m.*\d+s.*\d+(?:ms|s)/step.*$', 
        re.MULTILINE
    )
    cleaned_text = ansi_timing_pattern.sub('', cleaned_text)
    
    # More aggressive pattern that matches any line containing the progress bar and timing info
    # This is a fallback in case the specific patterns above miss something
    progress_pattern = re.compile(
        r'^.*\d+/\d+.*━+.*\d+s.*(?:ms|s)/step.*$',
        re.MULTILINE
    )
    cleaned_text = progress_pattern.sub('', cleaned_text)
    
    # Remove lines with backspace characters (^H) which are used for updating progress
    backspace_pattern = re.compile(r'^.*\^H.*$', re.MULTILINE)
    cleaned_text = backspace_pattern.sub('', cleaned_text)
    
    # Remove potential multiple consecutive empty lines left behind
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text).strip()
    
    return cleaned_text


def clean_log(log_text, remove_timing=True, remove_debug=True):
    """
    Comprehensive log cleaning function that can remove both timing logs and debug messages.
    
    Parameters
    ----------
    log_text : str
        The original log text to clean.
    remove_timing : bool, default=True
        Whether to remove SpliceAI timing information.
    remove_debug : bool, default=True
        Whether to remove debug messages.
        
    Returns
    -------
    str
        Cleaned log text with specified elements removed.
    """
    cleaned_text = log_text
    
    if remove_timing:
        cleaned_text = strip_spliceai_timing_logs(cleaned_text)
        
    if remove_debug:
        cleaned_text = strip_debug_messages(cleaned_text)
    
    return cleaned_text

# Command-line usage
if __name__ == '__main__':
    import sys
    import os
    import argparse
    
    # Set up command line arguments with argparse for better user interface
    parser = argparse.ArgumentParser(description="Clean log files by removing timing information and debug messages")
    parser.add_argument("input_file", help="Input log file to process")
    parser.add_argument("-o", "--output", help="Output file (default: input_file.clean.log)")
    parser.add_argument("--keep-timing", action="store_true", help="Keep SpliceAI timing information (default: remove)")
    parser.add_argument("--keep-debug", action="store_true", help="Keep debug messages (default: remove)")
    parser.add_argument("--timing-only", action="store_true", help="Only remove timing information, keep debug messages")
    parser.add_argument("--debug-only", action="store_true", help="Only remove debug messages, keep timing information")
    
    args = parser.parse_args()
    
    # Determine what to remove based on arguments
    remove_timing = not args.keep_timing and not args.debug_only
    remove_debug = not args.keep_debug and not args.timing_only
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}.clean{ext}"
    
    print(f"Processing {args.input_file}...")
    print(f"Removing: {' timing' if remove_timing else ''}{' debug messages' if remove_debug else ''}")
    
    try:
        # Read input file
        with open(args.input_file, 'r', errors='replace') as f:
            original_log = f.read()
        
        # Clean the log using our comprehensive function
        cleaned_log = clean_log(original_log, remove_timing=remove_timing, remove_debug=remove_debug)
        
        # Write output
        with open(output_file, 'w') as f:
            f.write(cleaned_log)
            
        # Report stats
        original_lines = original_log.count('\n')
        cleaned_lines = cleaned_log.count('\n')
        removed_lines = original_lines - cleaned_lines
        reduction = (1 - len(cleaned_log) / len(original_log)) * 100
        
        print(f"Done! Removed approximately {removed_lines} lines ({reduction:.1f}% reduction)")
        print(f"Output written to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
