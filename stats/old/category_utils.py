"""Utility functions for handling disease category labels.

This module provides functions to correct misspellings and standardize
category names for display purposes across all plotting scripts.
"""

def correct_category_spelling(category):
    """Correct misspellings in category names for display purposes.
    
    The source data contains 'Muscloskeletal' (misspelled), but we want
    to display it correctly as 'Musculoskeletal'.
    
    Parameters
    ----------
    category : str
        The category name from the source data
        
    Returns
    -------
    str
        The corrected category name for display
    """
    if category == 'Muscloskeletal':
        return 'Musculoskeletal'
    return category


def correct_category_dataframe(df, category_column='Category'):
    """Apply spelling corrections to a category column in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing category labels
    category_column : str, default='Category'
        Name of the column containing category labels
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with corrected category labels (modified in place)
    """
    if category_column in df.columns:
        df[category_column] = df[category_column].apply(correct_category_spelling)
    return df


def correct_category_list(categories):
    """Apply spelling corrections to a list of category names.
    
    Parameters
    ----------
    categories : list
        List of category names
        
    Returns
    -------
    list
        List with corrected category names
    """
    return [correct_category_spelling(cat) for cat in categories]
